import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from project_W_runner.utils import prepare_audio, transcribe

from .logger import get_logger

logger = get_logger("project-W-runner")

# heartbeat interval in seconds.
# this should be well below the heartbeat
# timeout of the server.
HEARTBEAT_INTERVAL = 10
HEARTBEAT_TIMEOUT = 60


@dataclass
class JobData:
    """
    Data for the current job.
    This only contains data relevant for the runner,
    so no metadata like filename, job id or job owner.
    """

    id: Optional[int]
    audio: Optional[bytes]
    model: Optional[str]
    language: Optional[str]
    transcript: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    current_step: Optional[str] = None


class ShutdownSignal(Exception):
    """
    Exception that is raised when the runner should shut down.
    This could be either because a graceful shutdown was requested
    or because of an error, e.g. if a server request times out.
    """

    reason: str

    def __init__(self, reason: str):
        self.reason = reason


class Runner:
    backend_url: str
    token: str
    torch_device: Optional[str]
    model_cache_dir: Optional[str]
    id: Optional[int]
    current_job_data: Optional[JobData]  # protected by the following cond element
    current_job_data_cond: asyncio.Condition
    current_job_aborted: bool
    commandThreadToExit: bool  # to signal threads that they should stop
    session: aiohttp.ClientSession

    def __init__(
        self,
        backend_url: str,
        token: str,
        torch_device: Optional[str],
        model_cache_dir: Optional[str] = None,
    ):
        self.backend_url = backend_url
        self.token = token
        self.torch_device = torch_device
        self.model_cache_dir = model_cache_dir
        self.commandThreadToExit = False
        self.current_job_data = None
        self.current_job_data_cond = asyncio.Condition()
        self.current_job_result = None
        self.current_job_aborted = False

    async def getJobAudio(self) -> tuple[bytes | None, dict | None, int]:
        """
        Get the binary data of the audio file from /api/runners/retrieveJobAudio route
        """
        headers = {"Authorization": f"Bearer {self.token}"}
        async with self.session.get(
            self.backend_url + "/api/runners/retrieveJobAudio", headers=headers
        ) as response:
            if response.content_type == "audio/basic":
                return await response.read(), None, response.status
            if response.content_type == "application/json":
                return None, await response.json(), response.status
            return (
                None,
                {
                    "error": f"Backend response is neither json nor basic audio but of type {response.content_type}"
                },
                400,
            )

    async def post(
        self,
        route: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        append_auth_header: bool = True,
    ) -> tuple[dict, int]:
        """
        Send a POST request to the server.

        Optionally accepts `data` (a dictionary of form data) and/or `params` (a dictionary of query parameters).

        If `append_auth_header` is True (which it is by default), the runner's token is appended to the request headers.
        """
        headers = {"Authorization": f"Bearer {self.token}"} if append_auth_header else {}
        async with self.session.post(
            self.backend_url + route, data=data, params=params, headers=headers
        ) as response:
            if response.content_type == "application/json":
                return await response.json(), response.status
            return {"error": f"Non-JSON backend response of type {response.content_type}"}, 400

    async def register(self):
        """
        Register the runner with the server.
        This should be called before the main loop or
        if the server requests a re-registration.
        """
        res, status = await self.post("/api/runners/register")
        # specifically also crash if the runner is already registered to prevent multiple runners with the same token to talk with the backend
        if status != 200:
            raise ShutdownSignal(f"Failed to register runner: {res.get('error')}")
        self.id = res.get("runnerID")
        logger.info(f"Runner registered, this runner has ID {self.id}")

    async def unregister(self):
        """
        Unregister the runner with the server.
        This should be called before ending the program, maybe in a finally clause
        """
        res, status = await self.post("/api/runners/unregister")
        if status != 200:
            # don't throw ShutdownSignal here because unregister is only called when the runner is already shutting down
            logger.warning(f"Failed to unregister runner: {res.get('error')}")
            return
        logger.info("Runner unregistered")

    def process_job(self, job_data: JobData) -> str:
        """
        Processes the current job, using the Whisper python package.
        This needs to run in a thread since this executes a lot of blocking tasks
        The calling task needs to hold current_job_data_cond on behalf of this thread because asyncio synchronization primitives are not thread save!
        """
        assert job_data.audio != None

        # For some silly reason python doesn't let you do assignments in a lambda.
        def progress_callback(progress: float):
            assert self.current_job_data != None
            self.current_job_data.progress = progress
            logger.info(f"Progress: {round(progress * 100, 2)}%")
            if self.commandThreadToExit:
                self.commandThreadToExit = False
                raise ShutdownSignal(
                    "progress_callback received signal to shutdown the processing thread"
                )

        audio = prepare_audio(job_data.audio)
        result = transcribe(
            audio,
            job_data.model,
            job_data.language,
            progress_callback,
            self.torch_device,
            self.model_cache_dir,
        )

        return result["text"]

    def stop_processing(self):
        if not self.commandThreadToExit:
            logger.info("Shutting down the processing thread...")
            self.commandThreadToExit = True

    def abort_job(self):
        if not self.current_job_aborted and self.current_job_data != None:
            logger.info("Received request to abort current job")
            self.current_job_aborted = True
            self.stop_processing()

    async def job_handler_task(self):
        """
        This task waits for a new task being assigned to this runner (gets notified by heartbeat_task).
        Then it will retrieve the job, create a new task to process it, waits for it to finish or to be aborted and then processes and submits the result
        """
        while True:
            async with self.current_job_data_cond:
                await self.current_job_data_cond.wait()  # waits for heartbeat_task to notify this task

                # retrieve this new job
                infoRes, infoCode = await self.post("/api/runners/retrieveJobInfo")
                if infoCode != 200:
                    raise ShutdownSignal(f"Failed to retrieve job info: {infoRes.get('error')}")
                audioRes, audioJson, audioCode = await self.getJobAudio()
                if audioCode != 200 or audioRes == None:
                    raise ShutdownSignal(
                        f"Failed to retrieve job audio: {audioJson.get('error') if audioJson != None else 'response not json'}"
                    )

                self.current_job_data = JobData(
                    id=infoRes.get("jobID"),
                    audio=audioRes,
                    model=infoRes.get("model"),
                    language=infoRes.get("language"),
                )
                logger.info(f"Job with ID {self.current_job_data.id} retrieved")

            # process job
            try:
                transcript = await asyncio.to_thread(self.process_job, self.current_job_data)
                self.current_job_data.transcript = transcript
            except ShutdownSignal as e:
                if self.current_job_aborted:
                    self.current_job_data.error = "job was aborted"
                    logger.info(f"Job with ID {self.current_job_data.id} was aborted")
                else:
                    raise e
            except Exception as e:
                logger.error(f"Error processing job {self.current_job_data.id}: {e}")
                self.current_job_data.error = str(e)

            # Submit the result to the server.
            async with self.current_job_data_cond:
                data = {}
                if self.current_job_data.transcript is not None:
                    data["transcript"] = self.current_job_data.transcript
                elif self.current_job_data.error is not None:
                    data["error_msg"] = self.current_job_data.error
                else:
                    # Sanity check if somehow neither transcript nor error is set.
                    data["error_msg"] = "Unknown runner error"
                    logger.error("Unknown error occurred while processing job")

                res, status = await self.post("/api/runners/submitJobResult", data=data)
                if status != 200:
                    raise ShutdownSignal(
                        f"Failed to submit job {self.current_job_data.id}: {res.get('error')}"
                    )
                logger.info(f"Result of job {self.current_job_data.id} submitted to backend")

                self.current_job_aborted = False
                self.current_job_data = None

    async def heartbeat_task(self):
        """
        Send a heartbeat to the server at regular intervals.
        This also recognizes when the server dispatched a new job
        to this runner, and initiates the job processing.
        If the server unregisters the runner, this function will return.
        Otherwise, it will keep sending heartbeats until shutdown.
        """
        while True:
            await self.register()
            timestamp_of_last_heartbeat = time.time()
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                data = {}
                if self.current_job_data and self.current_job_data.progress is not None:
                    data["progress"] = self.current_job_data.progress
                res, status = await self.post("/api/runners/heartbeat", data=data)
                if res.get("error") == "This runner is not currently registered as online!":
                    logger.warning(
                        "This runner was unexpectedly not registered anymore. Trying to re-register now..."
                    )
                    break
                if status != 200:
                    # The heartbeat failed for some reason. We don't want the runner
                    # to crash yet, so just continue the loop and try again in the next iteration if HEARTBEAT_TIMEOUT seconds haven't passed yet
                    if (time.time() - timestamp_of_last_heartbeat) < (
                        HEARTBEAT_TIMEOUT - HEARTBEAT_INTERVAL
                    ):
                        logger.warning(f"Heartbeat failed! Retrying in next iteration...")
                        continue
                    else:
                        raise ShutdownSignal(
                            f"Heartbeat kept failing for too long: {res.get('error')}: {res.get('msg')}"
                        )
                else:
                    timestamp_of_last_heartbeat = time.time()

                if self.current_job_data is None:
                    if "jobAssigned" in res:
                        logger.info("Job assigned")
                        # Before fetching the actual job data, make sure that the field
                        # is not None, so that we don't process the same job twice.
                        # This is also expected to be the case by the job processing tasks
                        async with self.current_job_data_cond:
                            self.current_job_data = JobData(
                                id=None, audio=None, model=None, language=None
                            )
                            # notify job_handler_task about new job
                            self.current_job_data_cond.notify()

                elif res.get("abort"):
                    self.abort_job()

    async def run(self):
        # Store the session so we can use it in other methods. Note that
        # we only exit this context manager when the runner is shutting down,
        # at which point we're not making any more requests over this session.
        async with aiohttp.ClientSession() as self.session:
            # create the two main tasks in a TaskGroup. This has the benefit that if any of these tasks raise an exception all tasks end and not just that one
            # all of these tasks need to stay active for the runner to work, if any of them crashes we want the runner to be restarted by docker or whatever to get to a working state again
            # otherwise the runner would just stay in a broken state forever until restarted manually
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.heartbeat_task())
                    tg.create_task(self.job_handler_task())
            except ExceptionGroup as e:
                if len(e.exceptions) == 1 and isinstance(e.exceptions[0], ShutdownSignal):
                    logger.fatal(f"Shutting down: {e.exceptions[0].reason}")
                else:
                    raise e
            finally:
                self.stop_processing()
                await self.unregister()

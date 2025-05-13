import asyncio
import json
import time
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper

import aiohttp
from pydantic import HttpUrl

from ._version import __version__, __version_tuple__
from .logger import get_logger
from .models.internal import JobData
from .models.request_data import (
    HeartbeatRequest,
    RunnerRegisterRequest,
    RunnerSubmitResultRequest,
    Transcript,
)
from .models.response_data import HeartbeatResponse, RunnerJobInfoResponse
from .models.settings import Settings
from .utils import transcribe

logger = get_logger("project-W-runner")

# heartbeat interval in seconds.
# this should be well below the heartbeat
# timeout of the server.
HEARTBEAT_INTERVAL = 10
HEARTBEAT_TIMEOUT = 60


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
    config: Settings
    backend_url: str
    source_code_url = "https://github.com/JulianFP/project-W-runner"
    id: int | None
    job_tmp_file: _TemporaryFileWrapper
    current_job_data: JobData | None  # protected by the following cond element
    current_job_data_cond: asyncio.Condition
    current_job_aborted: bool
    command_thread_to_exit: bool  # to signal threads that they should stop
    session: aiohttp.ClientSession

    def __init__(
        self,
        config: Settings,
        backend_url: HttpUrl,
    ):
        self.config = config
        self.backend_url = str(backend_url)
        if self.backend_url[-1] != "/":
            self.backend_url += "/"
        self.command_thread_to_exit = False
        self.current_job_data = None
        self.current_job_data_cond = asyncio.Condition()
        self.current_job_result = None
        self.current_job_aborted = False

    def __get_error_from_res(self, res: dict | str) -> str:
        return str(res.get("detail")) if type(res) == dict else str(res)

    async def get_job_audio(self):
        """
        Get the binary data of the audio file from /runners/retrieve_job_audio route
        """
        headers = {"Authorization": f"Bearer {self.config.runner_token}"}
        async with self.session.get(
            self.backend_url + "runners/retrieve_job_audio", headers=headers
        ) as response:
            base_mime_type = response.content_type.split("/")[0].strip()
            if base_mime_type in ["audio", "video"]:
                async for chunk in response.content.iter_chunked(10240):
                    self.job_tmp_file.write(chunk)
            elif response.content_type == "application/json":
                response_json = await response.json()
                raise ShutdownSignal(
                    f"Error while trying to retrieve job audio: {self.__get_error_from_res(response_json)}"
                )
            else:
                raise ShutdownSignal(f"Error while trying to retrieve job audio: invalid response")

    async def post(
        self,
        route: str,
        data: dict | None = None,
        params: dict | None = None,
        append_auth_header: bool = True,
    ) -> tuple[dict | str, int]:
        """
        Send a POST request to the server.

        Optionally accepts `data` (a dictionary of form data) and/or `params` (a dictionary of query parameters).

        If `append_auth_header` is True (which it is by default), the runner's token is appended to the request headers.
        """
        headers = (
            {"Authorization": f"Bearer {self.config.runner_token}"} if append_auth_header else {}
        )
        async with self.session.post(
            self.backend_url + route,
            data=json.dumps(data),
            params=params,
            headers=headers,
        ) as response:
            if response.content_type == "application/json":
                return await response.json(), response.status
            else:
                return await response.text(), response.status

    async def register(self):
        """
        Register the runner with the server.
        This should be called before the main loop or
        if the server requests a re-registration.
        """
        res, status = await self.post(
            "runners/register",
            data=RunnerRegisterRequest(
                name=self.config.runner_name,
                priority=self.config.runner_priority,
                version=__version__,
                git_hash=str(__version_tuple__[3]),
                source_code_url=self.source_code_url,
            ).model_dump(),
        )
        # specifically also crash if the runner is already registered to prevent multiple runners with the same token to talk with the backend
        if status >= 400 or type(res) != str:
            raise ShutdownSignal(f"Failed to register runner: {self.__get_error_from_res(res)}")
        self.id = int(res)
        logger.info(f"Runner registered, this runner has ID {self.id}")

    async def unregister(self):
        """
        Unregister the runner with the server.
        This should be called before ending the program, maybe in a finally clause
        """
        res, status = await self.post("runners/unregister")
        if status >= 400:
            # don't throw ShutdownSignal here because unregister is only called when the runner is already shutting down
            logger.warning(f"Failed to unregister runner: {self.__get_error_from_res(res)}")
            return
        logger.info("Runner unregistered")

    def process_job(self, job_data: JobData) -> Transcript:
        """
        Processes the current job, using the Whisper python package.
        This needs to run in a thread since this executes a lot of blocking tasks
        The calling task needs to hold current_job_data_cond on behalf of this thread because asyncio synchronization primitives are not thread save!
        """
        assert job_data.settings != None

        # For some silly reason python doesn't let you do assignments in a lambda.
        def progress_callback(progress: float):
            assert self.current_job_data != None
            self.current_job_data.progress = progress
            logger.info(f"Progress: {round(progress * 100, 2)}%")
            if self.command_thread_to_exit:
                self.command_thread_to_exit = False
                raise ShutdownSignal(
                    "progress_callback received signal to shutdown the processing thread"
                )

        result = transcribe(
            self.job_tmp_file.name,
            job_data.settings,
            self.config.whisper_settings,
            progress_callback,
        )

        transcript_dict: dict[str, str | dict] = {}
        for key, val in result.items():
            transcript_dict["as_" + key] = val.getvalue()

        return Transcript.model_validate(transcript_dict)

    def stop_processing(self):
        if not self.command_thread_to_exit:
            logger.info("Shutting down the processing thread...")
            self.command_thread_to_exit = True

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
                info_res, info_code = await self.post("runners/retrieve_job_info")
                if info_code >= 400 or type(info_res) != dict:
                    raise ShutdownSignal(
                        f"Failed to retrieve job info: {self.__get_error_from_res(info_res)}"
                    )
                job_info = RunnerJobInfoResponse.model_validate(info_res)
                await self.get_job_audio()

                self.current_job_data = JobData(
                    id=job_info.id,
                    settings=job_info,
                )
                logger.info(f"Job with ID {self.current_job_data.id} retrieved")

            # process job
            try:
                transcript = await asyncio.to_thread(self.process_job, self.current_job_data)
                self.current_job_data.transcript = transcript
            except ShutdownSignal as e:
                if self.current_job_aborted:
                    self.current_job_data.error_msg = "job was aborted"
                    logger.info(f"Job with ID {self.current_job_data.id} was aborted")
                else:
                    raise e
            except Exception as e:
                logger.error(f"Error processing job {self.current_job_data.id}: {e}")
                self.current_job_data.error_msg = str(e)

            # Submit the result to the server.
            async with self.current_job_data_cond:
                data = RunnerSubmitResultRequest()
                if self.current_job_data.transcript is not None:
                    data.transcript = self.current_job_data.transcript
                elif self.current_job_data.error_msg is not None:
                    data.error_msg = self.current_job_data.error_msg
                else:
                    # Sanity check if somehow neither transcript nor error is set.
                    data.error_msg = "Unknown runner error"
                    logger.error("Unknown error occurred while processing job")

                res, status = await self.post("runners/submit_job_result", data=data.model_dump())
                if status >= 400:
                    raise ShutdownSignal(
                        f"Failed to submit job {self.current_job_data.id}: {self.__get_error_from_res(res)}"
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
                data = HeartbeatRequest()
                if self.current_job_data and self.current_job_data.progress is not None:
                    data.progress = self.current_job_data.progress
                res, status = await self.post("runners/heartbeat", data=data.model_dump())
                if (
                    self.__get_error_from_res(res)
                    == "This runner is not currently registered as online!"
                ):
                    logger.warning(
                        "This runner was unexpectedly not registered anymore. Trying to re-register now..."
                    )
                    break
                if status >= 400 or type(res) != dict:
                    # The heartbeat failed for some reason. We don't want the runner
                    # to crash yet, so just continue the loop and try again in the next iteration if HEARTBEAT_TIMEOUT seconds haven't passed yet
                    if (time.time() - timestamp_of_last_heartbeat) < (
                        HEARTBEAT_TIMEOUT - HEARTBEAT_INTERVAL
                    ):
                        logger.warning(f"Heartbeat failed! Retrying in next iteration...")
                        continue
                    else:
                        raise ShutdownSignal(
                            f"Heartbeat kept failing for too long: {self.__get_error_from_res(res)}: {str(res)}"
                        )
                else:
                    timestamp_of_last_heartbeat = time.time()
                    heartbeat_resp = HeartbeatResponse.model_validate(res)

                if self.current_job_data is None:
                    if heartbeat_resp.job_assigned:
                        logger.info("Job assigned")
                        # Before fetching the actual job data, make sure that the field
                        # is not None, so that we don't process the same job twice.
                        # This is also expected to be the case by the job processing tasks
                        async with self.current_job_data_cond:
                            self.current_job_data = JobData()
                            # notify job_handler_task about new job
                            self.current_job_data_cond.notify()

                elif heartbeat_resp.abort:
                    self.abort_job()

    async def run(self):
        # Store the session so we can use it in other methods. Note that
        # we only exit this context manager when the runner is shutting down,
        # at which point we're not making any more requests over this session.
        async with aiohttp.ClientSession() as self.session:
            with NamedTemporaryFile("wb", delete_on_close=False) as self.job_tmp_file:
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

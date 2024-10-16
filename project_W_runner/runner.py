import asyncio
import base64
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


@dataclass
class JobData:
    """
    Data for the current job.
    This only contains data relevant for the runner,
    so no metadata like filename, job id or job owner.
    """

    audio: bytes
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
    current_job_data: Optional[JobData]
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
        self.current_job_data = None
        self.job_task = None

    def process_current_job(self, job_data: JobData) -> str:
        """
        Processes the current job, using the Whisper python package.
        """

        # For some silly reason python doesn't let you do assignments in a lambda.
        def progress_callback(progress: float):
            logger.info(f"Progress: {round(progress * 100, 2)}%")
            self.current_job_data.progress = progress

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

    async def post(
        self, route: str, data: dict = None, params: dict = None, append_auth_header: bool = True
    ):
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
            logger.warning(f"Non-JSON backend response of type {response.content_type}")
            return None, response.status

    async def register(self):
        """
        Register the runner with the server.
        This should be called before the main loop or
        if the server requests a re-registration.
        """
        res, status = await self.post("/api/runners/register")
        if status != 200:
            raise ShutdownSignal(f"Failed to register runner: {res['error']}")
        logger.info("Runner registered")

    async def dispatch_job(self):
        """
        Retrieves a new job from the server and processes it in a background
        thread. Once the job is processed, the result is submitted to the server.
        This is called when the server assigns a new job to this runner.
        """
        res, code = await self.post("/api/runners/retrieveJob")
        if code != 200:
            raise ShutdownSignal(f"Failed to retrieve job: {res['error']}")
        self.current_job_data = JobData(
            audio=base64.b64decode(res["audio"]),
            model=res.get("model"),
            language=res.get("language"),
        )

        cancelled = False
        try:
            # Note: In order for the background thread to not block the heartbeat loop, it
            # must not keep the GIL throughout its execution. This is true in this case, because
            # the background thread mostly only does three things:
            # 1. Run ffmpeg in a subprocess (IO-bound)
            # 2. Download the whisper model (IO-bound)
            # 3. Run the whisper model (the GIL is released inside pytorch functions)
            transcript = await asyncio.to_thread(self.process_current_job, self.current_job_data)
            self.current_job_data.transcript = transcript
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            self.current_job_data.error = str(e)
        except asyncio.CancelledError:
            cancelled = True
            errorMsg = "job was aborted"
            logger.error(errorMsg)
            self.current_job_data.error = errorMsg

        finally:
            # Submit the result to the server.
            data = {}
            if self.current_job_data.transcript is not None:
                data["transcript"] = self.current_job_data.transcript
            elif self.current_job_data.error is not None:
                data["error_msg"] = self.current_job_data.error
            # Sanity check if somehow neither transcript nor error is set.
            if not data:
                data = {"error_msg": "Unexpected runner error!"}
            self.current_job_data = None
            res, status = await self.post("/api/runners/submitJobResult", data=data)
            if status != 200:
                raise ShutdownSignal(f"Failed to submit job: {res['error']}")
            # it is best to not swallow CancelledError. See https://docs.python.org/3/library/asyncio-task.html#task-cancellation
            if cancelled:
                raise asyncio.CancelledError

    async def heartbeat_loop(self):
        """
        Send a heartbeat to the server at regular intervals.
        This also recognizes when the server dispatched a new job
        to this runner, and initiates the job processing.
        If the server unregisters the runner, this function will return.
        Otherwise, it will keep sending heartbeats until shutdown.
        """
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            data = {}
            if self.current_job_data and self.current_job_data.progress is not None:
                data["progress"] = self.current_job_data.progress
            res, status = await self.post("/api/runners/heartbeat", data=data)
            if res is None:
                # The heartbeat didn't return JSON for some reason. We don't want the runner
                # to crash yet, so just continue the loop and try again in the next iteration.
                logger.warning(f"Heartbeat failed!")
                continue
            # If the server unregisterd the runner, we don't want to send more heartbeats,
            # so we exit the loop and let the run() method handle the re-registration.
            # TODO: Replace this with a proper error code check.
            if res.get("error") == "This runner is not currently registered as online!":
                return

            if status != 200:
                raise ShutdownSignal(f"Heartbeat failed: {res['msg']}")

            if self.current_job_data is None:
                if "jobAssigned" in res:
                    logger.info("Job assigned")
                    # Before fetching the actual job data, make sure that the field
                    # is not None, so that we don't process the same job twice.
                    self.current_job_data = JobData(audio=None, model=None, language=None)
                    # Start the job processing in the background. The task is stored
                    # in a field, because the event loop only keeps a weak reference
                    # to it, so it may get garbage collected if we don't store it.
                    self.job_task = asyncio.create_task(self.dispatch_job())
            elif res.get("error") == "Current job was aborted":
                if self.job_task is not None:
                    self.job_task.cancel()
                else:
                    self.current_job_data = None

    async def run(self):
        async with aiohttp.ClientSession() as session:
            # Store the session so we can use it in other methods. Note that
            # we only exit this context manager when the runner is shutting down,
            # at which point we're not making any more requests over this session.
            try:
                self.session = session
                while True:
                    await self.register()
                    # If this function returns, the runner was unregistered. It'll
                    # automatically re-register in the next iteration of the loop.
                    await self.heartbeat_loop()
            except ShutdownSignal as s:
                # If the server requested a shutdown, we don't want to re-register.
                logger.fatal(f"Shutting down: {s.reason}")
            finally:
                if self.job_task is not None:
                    self.job_task.cancel()
                await self.post("/api/runners/unregister")

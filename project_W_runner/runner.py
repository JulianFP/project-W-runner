import asyncio
import base64
from dataclasses import dataclass
from threading import Condition, Thread
import time
from typing import Optional

from project_W_runner.utils import prepare_audio, transcribe
from .logger import get_logger
import aiohttp

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
    current_job_data: Optional[JobData]
    session: aiohttp.ClientSession
    # We use this condition variable to signal to the processing thread
    # that a new job has been assigned and it should start processing it.
    cond_var: Condition
    new_job: bool = False

    def __init__(self, backend_url: str, token: str, torch_device: Optional[str]):
        self.backend_url = backend_url
        self.token = token
        self.torch_device = torch_device
        self.current_job_data = None
        self.cond_var = Condition()
        Thread(target=self.run_processing_thread, daemon=True).start()

    def run_processing_thread(self):
        # Note that this thread runs indefinitely and will only exit when the
        # program exits. But since the runner lives until program exit anyways,
        # this shouldn't matter.
        while True:
            with self.cond_var:
                while not self.new_job:
                    self.cond_var.wait()
                self.new_job = False
            # We have a new job, process it.
            self.process_current_job()

    def process_current_job(self):
        """
        Processes the current job, using the Whisper python package.
        """

        # For some silly reason python doesn't let you do assignments in a lambda.
        def progress_callback(progress: float):
            logger.info(f"Progress: {round(progress * 100, 2)}%")
            self.current_job_data.progress = progress

        audio = prepare_audio(self.current_job_data.audio)
        result = transcribe(audio, self.current_job_data.model, self.current_job_data.language, progress_callback, self.torch_device)
        self.current_job_data.transcript = result["text"]

    async def post(self, route: str, data: dict = None, params: dict = None, append_auth_header: bool = True):
        """
        Send a POST request to the server.

        Optionally accepts `data` (a dictionary of form data) and/or `params` (a dictionary of query parameters).

        If `append_auth_header` is True (which it is by default), the runner's token is appended to the request headers.
        """
        headers = {"Authorization": f"Bearer {self.token}"} if append_auth_header else {}
        async with self.session.post(self.backend_url + route, data=data, params=params, headers=headers) as response:
            return await response.json(), response.status

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
        Retrieves a new job from the server and signals to the processing thread
        that it should start processing it. This is called when the server assigns
        a new job to this runner.
        """
        # This does block in an async function, but the only contention could be
        # with the processing thread, which only ever keeps the condvar for a very
        # short time, so this should be fine.
        with self.cond_var:
            # Before fetching the actual job data, make sure that the field
            # is not None, so that we don't process the same job twice.
            self.current_job_data = JobData(
                audio=None,
                model=None,
                language=None
            )
            res, code = await self.post("/api/runners/retrieveJob")
            if code != 200:
                raise ShutdownSignal(f"Failed to retrieve job: {res['error']}")
            self.current_job_data.audio = base64.b64decode(res["audio"])
            self.current_job_data.model = res.get("model")
            self.current_job_data.language = res.get("language")
            self.new_job = True
            logger.info("Job downloaded, processing...")
            self.cond_var.notify()

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
            # If the server unregisterd the runner, we don't want to send more heartbeats,
            # so we exit the loop and let the run() method handle the re-registration.
            if res.get("error") == "This runner is not currently registered as online!":
                return

            if status != 200:
                raise ShutdownSignal(f"Heartbeat failed: {res['msg']}")

            if self.current_job_data is None:
                if "jobAssigned" in res:
                    logger.info("Job assigned")
                    asyncio.create_task(self.dispatch_job())
            else:
                data = {}
                if self.current_job_data.transcript is not None:
                    data["transcript"] = self.current_job_data.transcript
                elif self.current_job_data.error is not None:
                    data["error_msg"] = self.current_job_data.error
                if data:
                    self.current_job_data = None
                    res, status = await self.post("/api/runners/submitJobResult", data=data)
                    if status != 200:
                        raise ShutdownSignal(f"Failed to submit job: {res['error']}")

    async def run(self):
        try:
            async with aiohttp.ClientSession() as session:
                # Store the session so we can use it in other methods. Note that
                # we only exit this context manager when the runner is shutting down,
                # at which point we're not making any more requests over this session.
                self.session = session
                while True:
                    await self.register()
                    # If this function returns, the runner was unregistered. It'll
                    # automatically re-register in the next iteration of the loop.
                    await self.heartbeat_loop()
        except ShutdownSignal as s:
            logger.fatal(f"Shutting down: {s.reason}")
            return

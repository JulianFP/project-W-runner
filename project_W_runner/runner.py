import asyncio
import json
import ssl
import time
from io import StringIO
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import Any, Callable, TypeVar

import certifi
import httpx
from pydantic import ValidationError

from project_W_runner.models.base import JobSettingsBase

from ._version import __version__, __version_tuple__
from .logger import get_logger
from .models.internal import JobData, ResponseNotJson, RunnerId, ShutdownSignal
from .models.request_data import (
    HeartbeatRequest,
    RunnerRegisterRequest,
    RunnerSubmitResultRequest,
    Transcript,
)
from .models.response_data import HeartbeatResponse, RunnerJobInfoResponse
from .models.settings import Settings, WhisperSettings

logger = get_logger("project-W-runner")

# heartbeat interval in seconds.
# this should be well below the heartbeat
# timeout of the server.
HEARTBEAT_INTERVAL = 10
HEARTBEAT_TIMEOUT = 60

PydanticModel = TypeVar("PydanticModel", covariant=True)


class Runner:
    transcribe: Callable[
        [str, JobSettingsBase, WhisperSettings, Callable[[float], None]], dict[str, StringIO]
    ]
    config: Settings
    backend_url: str
    source_code_url = "https://github.com/JulianFP/project-W-runner"
    id: int | None
    current_job_data: JobData | None  # protected by the following cond element
    current_job_data_cond: asyncio.Condition
    current_job_aborted: bool
    command_thread_to_exit: bool  # to signal threads that they should stop
    session: httpx.AsyncClient

    def __init__(
        self,
        transcribe_function,
        config: Settings,
    ):
        self.transcribe = transcribe_function
        self.config = config
        self.backend_url = str(config.backend_settings.url)
        if self.backend_url[-1] != "/":
            self.backend_url += "/"
        self.backend_url += "api/runners/"
        self.command_thread_to_exit = False
        self.current_job_data = None
        self.current_job_data_cond = asyncio.Condition()
        self.current_job_result = None
        self.current_job_aborted = False

    async def get_job_audio(self, job_tmp_file: _TemporaryFileWrapper):
        """
        Get the binary data of the audio file from api/runners/retrieve_job_audio route
        """
        headers = {
            "Authorization": f"Bearer {self.config.backend_settings.auth_token.get_secret_value()}"
        }
        response = await self.session.post(f"{self.backend_url}retrieve_job_audio", headers=headers)
        if response.status_code >= 400:
            if response.headers.get("Content-Type") == "application/json" and (
                detail := (response.json()).get("detail")
            ):
                raise ShutdownSignal(f"Error while trying to retrieve job audio: {detail}")
            else:
                raise ShutdownSignal(
                    f"Error while trying to retrieve job audio: Returned status code {response.status_code} without attached detail"
                )
        base_mime_type = response.headers.get("Content-Type").split("/")[0].strip()
        if base_mime_type in ["audio", "video"]:
            async for chunk in response.aiter_bytes(10240):
                job_tmp_file.write(chunk)
        else:
            raise ShutdownSignal(
                f"Error while trying to retrieve job audio: The content type of the response is neither audio nor video"
            )

    async def get(
        self,
        route: str,
        params: dict | None = None,
        append_auth_header: bool = True,
    ) -> Any:
        """
        Send a GET request to the server.
        Optionally accepts `params` (a dictionary of query parameters).
        If `append_auth_header` is True (which it is by default), the runner's token is appended to the request headers.
        """
        headers = (
            {
                "Authorization": f"Bearer {self.config.backend_settings.auth_token.get_secret_value()}"
            }
            if append_auth_header
            else {}
        )
        response = await self.session.get(
            self.backend_url + route,
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        if response.headers.get("Content-Type") == "application/json":
            return response.json()
        else:
            raise ResponseNotJson(
                f"The backend returned with content_type {response.headers.get("Content-Type")} on a get request on route {route} even though 'application/json' was expected"
            )

    async def post(
        self,
        route: str,
        data: dict | None = None,
        params: dict | None = None,
        append_auth_header: bool = True,
    ) -> Any:
        """
        Send a POST request to the server.
        Optionally accepts `data` (a dictionary that gets send as json in the body) and/or `params` (a dictionary of query parameters).
        If `append_auth_header` is True (which it is by default), the runner's token is appended to the request headers.
        """
        headers = (
            {
                "Authorization": f"Bearer {self.config.backend_settings.auth_token.get_secret_value()}"
            }
            if append_auth_header
            else {}
        )
        response = await self.session.post(
            self.backend_url + route,
            json=data,
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        if response.headers.get("Content-Type") == "application/json":
            return response.json()
        else:
            raise ResponseNotJson(
                f"The backend returned with content_type {response.headers.get("Content-Type")} on a post request on route {route} even though 'application/json' was expected"
            )

    async def get_validated(
        self,
        route: str,
        return_model: type[PydanticModel],
        params: dict | None = None,
        append_auth_header: bool = True,
    ) -> PydanticModel:
        response = await self.get(route, params, append_auth_header)
        if type(response) == dict:
            return return_model(**response)
        else:
            return return_model(**{"root": response})  # expects Pydantic RootModel

    async def post_validated(
        self,
        route: str,
        return_model: type[PydanticModel],
        data: dict | None = None,
        params: dict | None = None,
        append_auth_header: bool = True,
    ) -> PydanticModel:
        response = await self.post(route, data, params, append_auth_header)
        if type(response) == dict:
            return return_model(**response)
        else:
            return return_model(**{"root": response})  # expects Pydantic RootModel

    async def register(self):
        """
        Register the runner with the server.
        This should be called before the main loop or
        if the server requests a re-registration.
        """

        try:
            runner_id = await self.post_validated(
                "register",
                RunnerId,
                data=RunnerRegisterRequest(
                    name=self.config.runner_name,
                    priority=self.config.runner_priority,
                    version=__version__,
                    git_hash=str(__version_tuple__[-1]).split(".")[0].removeprefix("g"),
                    source_code_url=self.source_code_url,
                ).model_dump(),
            )
        except (httpx.HTTPError, ValidationError, ResponseNotJson) as e:
            raise ShutdownSignal(f"Failed to register runner: {str(e)}")
        self.id = runner_id.root
        logger.info(f"Runner registered, this runner has ID {self.id}")

    async def unregister(self):
        """
        Unregister the runner with the server.
        This should be called before ending the program, maybe in a finally clause
        """
        try:
            await self.post("unregister")
        except (httpx.HTTPError, ValidationError, ResponseNotJson) as e:
            # don't throw ShutdownSignal here because unregister is only called when the runner is already shutting down
            logger.warning(f"Failed to unregister runner: {str(e)}")
            return
        logger.info("Runner unregistered")

    def process_job(self, job_data: JobData, job_tmp_file: _TemporaryFileWrapper) -> Transcript:
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
            if self.command_thread_to_exit:
                self.command_thread_to_exit = False
                raise ShutdownSignal(
                    "progress_callback received signal to shutdown the processing thread"
                )

        result = self.transcribe(
            job_tmp_file.name,
            job_data.settings,
            self.config.whisper_settings,
            progress_callback,
        )

        transcript_dict: dict[str, str | dict] = {}
        for key, val in result.items():
            if key == "json":
                transcript_dict["as_json"] = json.loads(val.getvalue())
            else:
                transcript_dict[f"as_{key}"] = val.getvalue()

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
            with NamedTemporaryFile("wb", delete_on_close=False) as job_tmp_file:
                async with self.current_job_data_cond:
                    await self.current_job_data_cond.wait()  # waits for heartbeat_task to notify this task

                    # retrieve this new job
                    try:
                        job_info = await self.get_validated(
                            "retrieve_job_info", RunnerJobInfoResponse
                        )
                    except (httpx.HTTPError, ValidationError, ResponseNotJson) as e:
                        raise ShutdownSignal(f"Failed to retrieve job info: {str(e)}")
                    await self.get_job_audio(job_tmp_file)

                    self.current_job_data = JobData(
                        id=job_info.id,
                        settings=job_info.settings,
                    )
                    logger.info(f"Job with ID {self.current_job_data.id} retrieved")

                # process job
                try:
                    transcript = await asyncio.to_thread(
                        self.process_job, self.current_job_data, job_tmp_file
                    )
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
                if self.current_job_data.transcript is not None:
                    data = RunnerSubmitResultRequest(transcript=self.current_job_data.transcript)
                elif self.current_job_data.error_msg is not None:
                    data = RunnerSubmitResultRequest(error_msg=self.current_job_data.error_msg)
                else:
                    # Sanity check if somehow neither transcript nor error is set.
                    data = RunnerSubmitResultRequest(error_msg="Unknown runner error")
                    logger.error("Unknown error occurred while processing job")

                try:
                    await self.post("submit_job_result", data=data.model_dump())
                except (httpx.HTTPError, ValidationError, ResponseNotJson) as e:
                    raise ShutdownSignal(
                        f"Failed to submit job {self.current_job_data.id}: {str(e)}"
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
                try:
                    heartbeat_resp = await self.post_validated(
                        "heartbeat", HeartbeatResponse, data=data.model_dump()
                    )
                except (httpx.HTTPError, ValidationError, ResponseNotJson) as e:
                    # The heartbeat failed for some reason. We don't want the runner
                    # to crash yet, so just continue the loop and try again in the next iteration if HEARTBEAT_TIMEOUT seconds haven't passed yet
                    if (time.time() - timestamp_of_last_heartbeat) < (
                        HEARTBEAT_TIMEOUT - HEARTBEAT_INTERVAL
                    ):
                        logger.warning(f"Heartbeat failed: {str(e)}! Retrying in next iteration...")
                        continue
                    else:
                        raise ShutdownSignal(f"Heartbeat kept failing for too long: {str(e)}")
                timestamp_of_last_heartbeat = time.time()

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
        if self.config.backend_settings.ca_pem_file_path:
            cafile = str(self.config.backend_settings.ca_pem_file_path)
        else:
            cafile = certifi.where()
        ctx = ssl.create_default_context(cafile=cafile)
        # Store the session so we can use it in other methods. Note that
        # we only exit this context manager when the runner is shutting down,
        # at which point we're not making any more requests over this session.
        async with httpx.AsyncClient(verify=ctx) as self.session:
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

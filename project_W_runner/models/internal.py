from pydantic import BaseModel, Field, RootModel

from .base import JobSettingsBase
from .request_data import Transcript


class ResponseNotJson(Exception):
    """
    This gets raised if backend response with non-json response even though json was expected
    """

    pass


class ShutdownSignal(Exception):
    """
    Exception that is raised when the runner should shut down.
    This could be either because a graceful shutdown was requested
    or because of an error, e.g. if a server request times out.
    """

    reason: str

    def __init__(self, reason: str, e: Exception | None = None):
        self.reason = f"{reason}: {type(e).__name__}: '{str(e)}'"


class JobData(BaseModel):
    id: int | None = None
    error_msg: str | None = None
    transcript: Transcript | None = None
    progress: float | None = Field(
        ge=0.0,
        le=100.0,
        default=None,
    )
    settings: JobSettingsBase | None = None


class RunnerId(RootModel):
    root: int

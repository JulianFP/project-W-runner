from pydantic import Field

from .request_data import JobSettings, RunnerSubmitResultRequest


class JobData(RunnerSubmitResultRequest):
    id: int | None = None
    progress: float | None = Field(
        ge=0.0,
        le=1.0,
        default=None,
    )
    settings: JobSettings | None = None

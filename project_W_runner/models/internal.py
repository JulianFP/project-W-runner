from pydantic import BaseModel, Field

from .request_data import JobSettings, Transcript


class JobData(BaseModel):
    id: int | None = None
    error_msg: str | None = None
    transcript: Transcript | None = None
    progress: float | None = Field(
        ge=0.0,
        le=1.0,
        default=None,
    )
    settings: JobSettings | None = None

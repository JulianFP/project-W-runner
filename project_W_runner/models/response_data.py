from pydantic import BaseModel

from .request_data import JobSettings


class HeartbeatResponse(BaseModel):
    abort: bool = False
    job_assigned: bool = False


class RunnerJobInfoResponse(BaseModel):
    id: int
    settings: JobSettings


class JobInfoToRunner(BaseModel):
    id: int
    settings: JobSettings

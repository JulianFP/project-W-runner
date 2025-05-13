from pydantic import BaseModel

from .request_data import JobSettings


class HeartbeatResponse(BaseModel):
    abort: bool = False
    job_assigned: bool = False


class RunnerJobInfoResponse(JobSettings):
    id: int

"""Deployment Pydantic schemas"""

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.deployment import DeploymentStatus
from app.models.llm_model import BackendType


class DeploymentBase(BaseModel):
    """Base deployment schema"""

    name: str = Field(..., min_length=1, max_length=255)
    model_id: int
    worker_id: int
    backend: BackendType = BackendType.VLLM
    gpu_indexes: list[int] | None = None
    extra_params: dict | None = None


class DeploymentCreate(DeploymentBase):
    """Schema for creating a deployment"""

    pass


class DeploymentUpdate(BaseModel):
    """Schema for updating a deployment"""

    name: str | None = None
    backend: BackendType | None = None
    status: DeploymentStatus | None = None
    status_message: str | None = None
    container_id: str | None = None
    port: int | None = None
    gpu_indexes: list[int] | None = None
    extra_params: dict | None = None


class WorkerSummary(BaseModel):
    """Worker summary for deployment response"""

    id: int
    name: str
    address: str
    status: str

    class Config:
        from_attributes = True


class ModelSummary(BaseModel):
    """Model summary for deployment response"""

    id: int
    name: str
    model_id: str
    source: str

    class Config:
        from_attributes = True


class DeploymentResponse(DeploymentBase):
    """Schema for deployment response"""

    id: int
    status: str
    status_message: str | None = None
    container_id: str | None = None
    port: int | None = None
    created_at: datetime
    updated_at: datetime
    worker: WorkerSummary | None = None
    model: ModelSummary | None = None

    class Config:
        from_attributes = True


class DeploymentListResponse(BaseModel):
    """Schema for deployment list response"""

    items: list[DeploymentResponse]
    total: int


class DeploymentLogsResponse(BaseModel):
    """Schema for deployment logs response"""

    deployment_id: int
    logs: str

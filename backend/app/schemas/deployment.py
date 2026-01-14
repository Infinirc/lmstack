"""Deployment Pydantic schemas"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.deployment import DeploymentStatus
from app.models.llm_model import BackendType


class DeploymentBase(BaseModel):
    """Base deployment schema"""
    name: str = Field(..., min_length=1, max_length=255)
    model_id: int
    worker_id: int
    backend: BackendType = BackendType.VLLM
    gpu_indexes: Optional[list[int]] = None
    extra_params: Optional[dict] = None


class DeploymentCreate(DeploymentBase):
    """Schema for creating a deployment"""
    pass


class DeploymentUpdate(BaseModel):
    """Schema for updating a deployment"""
    name: Optional[str] = None
    backend: Optional[BackendType] = None
    status: Optional[DeploymentStatus] = None
    status_message: Optional[str] = None
    container_id: Optional[str] = None
    port: Optional[int] = None
    gpu_indexes: Optional[list[int]] = None
    extra_params: Optional[dict] = None


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
    status_message: Optional[str] = None
    container_id: Optional[str] = None
    port: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    worker: Optional[WorkerSummary] = None
    model: Optional[ModelSummary] = None

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

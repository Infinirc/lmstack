"""
Model Files Pydantic Schemas

These schemas represent a virtual view of model files on workers,
aggregated from deployment data. A model "exists" on a worker when
there is (or has been) a deployment for it.
"""

from typing import Optional
from pydantic import BaseModel


class ModelFileDeployment(BaseModel):
    """Summary of a deployment associated with a model file"""

    id: int
    name: str
    status: str
    port: Optional[int] = None


class ModelFileView(BaseModel):
    """
    Virtual model file view - aggregated from deployments.

    Represents the presence of a model on a specific worker,
    including all deployments that use this model on this worker.
    """

    model_id: int
    worker_id: int
    model_name: str  # Display name of the model
    model_source: str  # HuggingFace model ID
    worker_name: str  # Worker display name
    worker_address: str  # Worker network address
    status: str  # "downloading" | "starting" | "ready" | "stopped"
    download_progress: float  # Download progress percentage (0-100)
    deployment_count: int  # Total deployments of this model on this worker
    running_count: int  # Number of running deployments
    deployments: list[ModelFileDeployment]  # Deployment details


class ModelFileListResponse(BaseModel):
    """Response schema for listing model files"""

    items: list[ModelFileView]
    total: int

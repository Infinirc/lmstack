"""Pydantic models for LMStack Worker API.

Contains request/response models for the worker agent API endpoints.
"""

from typing import Optional

from pydantic import BaseModel

# =============================================================================
# Deployment Models
# =============================================================================


class DeployRequest(BaseModel):
    """Request to deploy a model."""

    deployment_id: int
    deployment_name: str
    image: str
    command: list[str]
    model_id: str
    gpu_indexes: list[int] = [0]
    environment: dict[str, str] = {}
    port: Optional[int] = None  # Optional fixed port


class StopRequest(BaseModel):
    """Request to stop a container."""

    container_id: str


class LogsRequest(BaseModel):
    """Request for container logs."""

    container_id: str
    tail: int = 100


# =============================================================================
# Image Management Models
# =============================================================================


class ImagePullRequest(BaseModel):
    """Request to pull an image."""

    image: str
    registry_auth: Optional[dict[str, str]] = None


class ImageBuildRequest(BaseModel):
    """Request to build an image."""

    dockerfile: str
    tag: str
    build_args: Optional[dict[str, str]] = None


class ImageDeleteRequest(BaseModel):
    """Request to delete an image."""

    image_id: str
    force: bool = False


# =============================================================================
# Container Management Models
# =============================================================================


class ContainerCreateRequest(BaseModel):
    """Request to create a container."""

    name: str
    image: str
    command: Optional[list[str]] = None
    entrypoint: Optional[list[str]] = None
    env: Optional[dict[str, str]] = None
    ports: Optional[list[dict]] = None
    volumes: Optional[list[dict]] = None
    gpu_ids: Optional[list[int]] = None
    restart_policy: str = "no"
    labels: Optional[dict[str, str]] = None
    cpu_limit: Optional[float] = None
    memory_limit: Optional[int] = None
    cap_add: Optional[list[str]] = None


class ContainerActionRequest(BaseModel):
    """Request for container lifecycle actions."""

    container_id: str
    timeout: int = 10


class ContainerExecRequest(BaseModel):
    """Request to exec command in container."""

    command: list[str]
    tty: bool = False
    privileged: bool = False
    user: Optional[str] = None
    workdir: Optional[str] = None
    env: Optional[list[str]] = None

"""Container Management API routes

Provides endpoints for managing Docker containers across workers.
All operations are proxied to the appropriate worker agent.
"""

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.worker import Worker

logger = logging.getLogger(__name__)
router = APIRouter()

# HTTP client timeouts
DEFAULT_TIMEOUT = 30.0
CONTAINER_ACTION_TIMEOUT = 60.0


# =============================================================================
# Request/Response Schemas
# =============================================================================


class PortMapping(BaseModel):
    """Container port mapping."""

    container_port: int
    host_port: int | None = None
    protocol: str = "tcp"


class VolumeMount(BaseModel):
    """Container volume mount."""

    source: str
    destination: str
    mode: str = "rw"


class ContainerCreateRequest(BaseModel):
    """Request to create a new container."""

    worker_id: int
    name: str
    image: str
    command: list[str] | None = None
    entrypoint: list[str] | None = None
    env: dict[str, str] | None = None
    ports: list[PortMapping] | None = None
    volumes: list[VolumeMount] | None = None
    gpu_ids: list[int] | None = None
    restart_policy: str = "no"
    labels: dict[str, str] | None = None
    cpu_limit: float | None = None
    memory_limit: int | None = None


class ContainerExecRequest(BaseModel):
    """Request to execute a command in a container."""

    command: list[str]
    tty: bool = False
    privileged: bool = False
    user: str | None = None
    workdir: str | None = None
    env: list[str] | None = None


class ContainerResponse(BaseModel):
    """Container information response."""

    id: str
    worker_id: int
    worker_name: str
    name: str
    image: str
    image_id: str
    state: str
    status: str
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    ports: list[dict[str, Any]] = []
    volumes: list[dict[str, Any]] = []
    gpu_ids: list[str] | None = None
    deployment_id: int | None = None
    deployment_name: str | None = None
    is_managed: bool = False
    env: list[str] | None = None


class ContainerStatsResponse(BaseModel):
    """Container resource usage stats."""

    cpu_percent: float
    memory_usage: int
    memory_limit: int
    memory_percent: float
    network_rx: int
    network_tx: int
    block_read: int
    block_write: int
    pids: int


class ContainerLogsResponse(BaseModel):
    """Container logs response."""

    container_id: str
    logs: str
    stdout: str | None = None
    stderr: str | None = None


class ContainerExecResponse(BaseModel):
    """Container exec command result."""

    exit_code: int
    stdout: str
    stderr: str


class ContainerListResponse(BaseModel):
    """Paginated container list response."""

    items: list[ContainerResponse]
    total: int


# =============================================================================
# Helper Functions
# =============================================================================


async def get_worker_or_404(
    worker_id: int,
    db: AsyncSession,
) -> Worker:
    """Get worker by ID or raise 404."""
    result = await db.execute(select(Worker).where(Worker.id == worker_id))
    worker = result.scalar_one_or_none()
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
    if worker.status != "online":
        raise HTTPException(
            status_code=400,
            detail=f"Worker {worker.name} is not online (status: {worker.status})",
        )
    return worker


async def call_worker_api(
    worker: Worker,
    method: str,
    path: str,
    timeout: float = DEFAULT_TIMEOUT,
    **kwargs,
) -> dict:
    """Call worker agent API and handle errors."""
    url = f"http://{worker.address}{path}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, **kwargs)

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Container not found on worker")

            if response.status_code >= 400:
                detail = response.json().get("detail", response.text)
                raise HTTPException(status_code=response.status_code, detail=detail)

            return response.json()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to worker {worker.name} at {worker.address}",
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Worker {worker.name} request timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error calling worker API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def container_to_response(container: dict, worker: Worker) -> ContainerResponse:
    """Convert worker container data to response model."""
    return ContainerResponse(
        id=container["id"],
        worker_id=worker.id,
        worker_name=worker.name,
        name=container["name"],
        image=container["image"],
        image_id=container.get("image_id", ""),
        state=container["state"],
        status=container.get("status", container["state"]),
        created_at=container.get("created_at"),
        started_at=container.get("started_at"),
        finished_at=container.get("finished_at"),
        exit_code=container.get("exit_code"),
        ports=container.get("ports", []),
        volumes=container.get("volumes", []),
        gpu_ids=container.get("gpu_ids"),
        deployment_id=container.get("deployment_id"),
        deployment_name=container.get("deployment_name"),
        is_managed=container.get("is_managed", False),
        env=container.get("env"),
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("", response_model=ContainerListResponse)
async def list_containers(
    worker_id: int | None = Query(None, description="Filter by worker ID"),
    all: bool = Query(True, description="Include stopped containers"),
    managed_only: bool = Query(False, description="Only LMStack-managed containers"),
    state: str | None = Query(None, description="Filter by state"),
    db: AsyncSession = Depends(get_db),
):
    """List containers across workers.

    If worker_id is provided, lists containers only from that worker.
    Otherwise, lists containers from all online workers.
    """
    all_containers: list[ContainerResponse] = []

    if worker_id:
        workers = [await get_worker_or_404(worker_id, db)]
    else:
        result = await db.execute(select(Worker).where(Worker.status == "online"))
        workers = list(result.scalars().all())

    for worker in workers:
        try:
            data = await call_worker_api(
                worker,
                "GET",
                "/containers",
                params={"all": all, "managed_only": managed_only},
            )

            for container in data.get("items", []):
                # Apply state filter
                if state and container.get("state") != state:
                    continue

                all_containers.append(container_to_response(container, worker))

        except HTTPException as e:
            if e.status_code != 503:
                logger.warning(f"Error listing containers from {worker.name}: {e.detail}")
        except Exception as e:
            logger.warning(f"Error listing containers from {worker.name}: {e}")

    return ContainerListResponse(items=all_containers, total=len(all_containers))


@router.get("/{container_id}", response_model=ContainerResponse)
async def get_container(
    container_id: str,
    worker_id: int = Query(..., description="Worker ID where the container is located"),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed information about a container."""
    worker = await get_worker_or_404(worker_id, db)
    data = await call_worker_api(worker, "GET", f"/containers/{container_id}")
    return container_to_response(data, worker)


@router.get("/{container_id}/stats", response_model=ContainerStatsResponse)
async def get_container_stats(
    container_id: str,
    worker_id: int = Query(..., description="Worker ID where the container is located"),
    db: AsyncSession = Depends(get_db),
):
    """Get container resource usage statistics."""
    worker = await get_worker_or_404(worker_id, db)
    return await call_worker_api(worker, "GET", f"/containers/{container_id}/stats")


@router.post("", response_model=ContainerResponse, status_code=201)
async def create_container(
    request: ContainerCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create and start a new container on a worker."""
    worker = await get_worker_or_404(request.worker_id, db)

    payload = {
        "name": request.name,
        "image": request.image,
    }

    # Add optional fields
    if request.command is not None:
        payload["command"] = request.command
    if request.entrypoint is not None:
        payload["entrypoint"] = request.entrypoint
    if request.env is not None:
        payload["env"] = request.env
    if request.ports is not None:
        payload["ports"] = [p.model_dump() for p in request.ports]
    if request.volumes is not None:
        payload["volumes"] = [v.model_dump() for v in request.volumes]
    if request.gpu_ids is not None:
        payload["gpu_ids"] = request.gpu_ids
    if request.restart_policy:
        payload["restart_policy"] = request.restart_policy
    if request.labels is not None:
        payload["labels"] = request.labels
    if request.cpu_limit is not None:
        payload["cpu_limit"] = request.cpu_limit
    if request.memory_limit is not None:
        payload["memory_limit"] = request.memory_limit

    data = await call_worker_api(
        worker,
        "POST",
        "/containers",
        json=payload,
        timeout=CONTAINER_ACTION_TIMEOUT,
    )

    return container_to_response(data, worker)


@router.post("/{container_id}/start", response_model=ContainerResponse)
async def start_container(
    container_id: str,
    worker_id: int = Query(..., description="Worker ID where the container is located"),
    db: AsyncSession = Depends(get_db),
):
    """Start a stopped container."""
    worker = await get_worker_or_404(worker_id, db)
    data = await call_worker_api(
        worker,
        "POST",
        f"/containers/{container_id}/start",
        timeout=CONTAINER_ACTION_TIMEOUT,
    )
    return container_to_response(data, worker)


@router.post("/{container_id}/stop", response_model=ContainerResponse)
async def stop_container(
    container_id: str,
    worker_id: int = Query(..., description="Worker ID where the container is located"),
    timeout: int = Query(10, description="Seconds to wait before killing"),
    db: AsyncSession = Depends(get_db),
):
    """Stop a running container."""
    worker = await get_worker_or_404(worker_id, db)
    data = await call_worker_api(
        worker,
        "POST",
        f"/containers/{container_id}/stop",
        params={"timeout": timeout},
        timeout=CONTAINER_ACTION_TIMEOUT,
    )
    return container_to_response(data, worker)


@router.post("/{container_id}/restart", response_model=ContainerResponse)
async def restart_container(
    container_id: str,
    worker_id: int = Query(..., description="Worker ID where the container is located"),
    timeout: int = Query(10, description="Seconds to wait during stop"),
    db: AsyncSession = Depends(get_db),
):
    """Restart a container."""
    worker = await get_worker_or_404(worker_id, db)
    data = await call_worker_api(
        worker,
        "POST",
        f"/containers/{container_id}/restart",
        params={"timeout": timeout},
        timeout=CONTAINER_ACTION_TIMEOUT,
    )
    return container_to_response(data, worker)


@router.post("/{container_id}/pause", response_model=ContainerResponse)
async def pause_container(
    container_id: str,
    worker_id: int = Query(..., description="Worker ID where the container is located"),
    db: AsyncSession = Depends(get_db),
):
    """Pause a running container."""
    worker = await get_worker_or_404(worker_id, db)
    data = await call_worker_api(
        worker,
        "POST",
        f"/containers/{container_id}/pause",
    )
    return container_to_response(data, worker)


@router.post("/{container_id}/unpause", response_model=ContainerResponse)
async def unpause_container(
    container_id: str,
    worker_id: int = Query(..., description="Worker ID where the container is located"),
    db: AsyncSession = Depends(get_db),
):
    """Unpause a paused container."""
    worker = await get_worker_or_404(worker_id, db)
    data = await call_worker_api(
        worker,
        "POST",
        f"/containers/{container_id}/unpause",
    )
    return container_to_response(data, worker)


@router.delete("/{container_id}", status_code=204)
async def delete_container(
    container_id: str,
    worker_id: int = Query(..., description="Worker ID where the container is located"),
    force: bool = Query(False, description="Force removal of running container"),
    volumes: bool = Query(False, description="Remove associated volumes"),
    db: AsyncSession = Depends(get_db),
):
    """Delete a container from a worker."""
    worker = await get_worker_or_404(worker_id, db)
    await call_worker_api(
        worker,
        "DELETE",
        f"/containers/{container_id}",
        params={"force": force, "volumes": volumes},
    )


@router.get("/{container_id}/logs", response_model=ContainerLogsResponse)
async def get_container_logs(
    container_id: str,
    worker_id: int = Query(..., description="Worker ID where the container is located"),
    tail: int = Query(100, ge=1, le=10000, description="Number of lines from end"),
    since: int | None = Query(None, description="Unix timestamp to start from"),
    until: int | None = Query(None, description="Unix timestamp to end at"),
    timestamps: bool = Query(True, description="Include timestamps"),
    db: AsyncSession = Depends(get_db),
):
    """Get container logs."""
    worker = await get_worker_or_404(worker_id, db)

    params = {
        "tail": tail,
        "timestamps": timestamps,
    }
    if since is not None:
        params["since"] = since
    if until is not None:
        params["until"] = until

    return await call_worker_api(
        worker,
        "GET",
        f"/containers/{container_id}/logs",
        params=params,
    )


@router.post("/{container_id}/exec", response_model=ContainerExecResponse)
async def exec_container_command(
    container_id: str,
    request: ContainerExecRequest,
    worker_id: int = Query(..., description="Worker ID where the container is located"),
    db: AsyncSession = Depends(get_db),
):
    """Execute a command in a running container."""
    worker = await get_worker_or_404(worker_id, db)

    payload = {
        "container_id": container_id,
        "command": request.command,
        "tty": request.tty,
        "privileged": request.privileged,
    }
    if request.user is not None:
        payload["user"] = request.user
    if request.workdir is not None:
        payload["workdir"] = request.workdir
    if request.env is not None:
        payload["env"] = request.env

    return await call_worker_api(
        worker,
        "POST",
        f"/containers/{container_id}/exec",
        json=payload,
        timeout=CONTAINER_ACTION_TIMEOUT,
    )

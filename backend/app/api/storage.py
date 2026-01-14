"""Storage Management API routes

Provides endpoints for managing Docker storage across workers.
Operations are proxied to the worker agent, or run locally for local workers.
"""

import logging

import docker
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.worker import Worker

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_TIMEOUT = 30.0


# =============================================================================
# Response Schemas
# =============================================================================


class StorageCategory(BaseModel):
    """Storage category stats."""

    count: int
    size: int
    reclaimable: int


class DiskUsageResponse(BaseModel):
    """Disk usage response."""

    worker_id: int
    worker_name: str
    images: StorageCategory
    containers: StorageCategory
    volumes: StorageCategory
    build_cache: StorageCategory
    total_size: int
    total_reclaimable: int


class VolumeResponse(BaseModel):
    """Volume information."""

    name: str
    worker_id: int
    worker_name: str
    driver: str
    mountpoint: str
    created_at: str
    labels: dict
    scope: str


class PruneRequest(BaseModel):
    """Prune request options."""

    images: bool = True
    containers: bool = True
    volumes: bool = False
    build_cache: bool = True


class PruneResponse(BaseModel):
    """Prune result."""

    worker_id: int
    worker_name: str
    images_deleted: int
    containers_deleted: int
    volumes_deleted: int
    build_cache_deleted: int
    space_reclaimed: int


# =============================================================================
# Helper Functions
# =============================================================================


def _is_local_worker(worker: Worker) -> bool:
    """Check if worker is a local worker (no agent)."""
    if not worker.address:
        return True
    host = worker.address.split(":")[0].lower()
    # Local workers have address "localhost" without port, or labels with type=local
    labels = worker.labels or {}
    if labels.get("type") == "local":
        return True
    # If address is just "localhost" without port, it's a local worker
    if host in ("localhost", "127.0.0.1", "local") and ":" not in worker.address:
        return True
    return False


async def get_worker(db: AsyncSession, worker_id: int) -> Worker:
    """Get worker by ID."""
    result = await db.execute(select(Worker).where(Worker.id == worker_id))
    worker = result.scalar_one_or_none()
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
    if worker.status != "online":
        raise HTTPException(status_code=400, detail=f"Worker {worker.name} is offline")
    return worker


async def call_worker(
    worker: Worker,
    method: str,
    path: str,
    timeout: float = DEFAULT_TIMEOUT,
    **kwargs,
) -> dict:
    """Call worker agent API."""
    url = f"http://{worker.address}{path}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, **kwargs)

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
        raise HTTPException(
            status_code=504, detail=f"Request to worker {worker.name} timed out"
        )


# =============================================================================
# Local Docker Operations
# =============================================================================


def _get_local_disk_usage() -> dict:
    """Get Docker disk usage from local Docker daemon."""
    client = docker.from_env()
    df = client.df()

    # Calculate image stats
    images = df.get("Images", []) or []
    image_count = len(images)
    image_size = sum(img.get("Size", 0) or 0 for img in images)
    image_reclaimable = sum(
        (img.get("Size", 0) or 0) for img in images if img.get("Containers", 0) == 0
    )

    # Calculate container stats
    containers = df.get("Containers", []) or []
    container_count = len(containers)
    container_size = sum(c.get("SizeRw", 0) or 0 for c in containers)
    container_reclaimable = sum(
        (c.get("SizeRw", 0) or 0) for c in containers if c.get("State") != "running"
    )

    # Calculate volume stats
    volumes = df.get("Volumes", []) or []
    volume_count = len(volumes)
    volume_size = sum(v.get("UsageData", {}).get("Size", 0) or 0 for v in volumes)
    volume_reclaimable = sum(
        (v.get("UsageData", {}).get("Size", 0) or 0)
        for v in volumes
        if v.get("UsageData", {}).get("RefCount", 0) == 0
    )

    # Calculate build cache stats
    build_cache = df.get("BuildCache", []) or []
    cache_count = len(build_cache)
    cache_size = sum(b.get("Size", 0) or 0 for b in build_cache)
    cache_reclaimable = sum(
        (b.get("Size", 0) or 0) for b in build_cache if not b.get("InUse", False)
    )

    total_size = image_size + container_size + volume_size + cache_size
    total_reclaimable = (
        image_reclaimable
        + container_reclaimable
        + volume_reclaimable
        + cache_reclaimable
    )

    return {
        "images": {
            "count": image_count,
            "size": image_size,
            "reclaimable": image_reclaimable,
        },
        "containers": {
            "count": container_count,
            "size": container_size,
            "reclaimable": container_reclaimable,
        },
        "volumes": {
            "count": volume_count,
            "size": volume_size,
            "reclaimable": volume_reclaimable,
        },
        "build_cache": {
            "count": cache_count,
            "size": cache_size,
            "reclaimable": cache_reclaimable,
        },
        "total_size": total_size,
        "total_reclaimable": total_reclaimable,
    }


def _get_local_volumes() -> list[dict]:
    """List volumes from local Docker daemon."""
    client = docker.from_env()
    volumes = client.volumes.list()
    result = []
    for vol in volumes:
        attrs = vol.attrs
        result.append(
            {
                "name": vol.name,
                "driver": attrs.get("Driver") or "local",
                "mountpoint": attrs.get("Mountpoint") or "",
                "created_at": attrs.get("CreatedAt") or "",
                "labels": attrs.get("Labels") or {},
                "scope": attrs.get("Scope") or "local",
            }
        )
    return result


def _delete_local_volume(volume_name: str, force: bool = False) -> dict:
    """Delete a volume from local Docker daemon."""
    client = docker.from_env()
    volume = client.volumes.get(volume_name)
    volume.remove(force=force)
    return {"status": "deleted"}


def _prune_local_storage(
    images: bool, containers: bool, volumes: bool, build_cache: bool
) -> dict:
    """Prune local Docker resources."""
    client = docker.from_env()
    result = {
        "images_deleted": 0,
        "containers_deleted": 0,
        "volumes_deleted": 0,
        "build_cache_deleted": 0,
        "space_reclaimed": 0,
    }

    if containers:
        prune_result = client.containers.prune()
        result["containers_deleted"] = len(prune_result.get("ContainersDeleted") or [])
        result["space_reclaimed"] += prune_result.get("SpaceReclaimed", 0)

    if images:
        prune_result = client.images.prune(filters={"dangling": False})
        result["images_deleted"] = len(prune_result.get("ImagesDeleted") or [])
        result["space_reclaimed"] += prune_result.get("SpaceReclaimed", 0)

    if volumes:
        prune_result = client.volumes.prune()
        result["volumes_deleted"] = len(prune_result.get("VolumesDeleted") or [])
        result["space_reclaimed"] += prune_result.get("SpaceReclaimed", 0)

    if build_cache:
        try:
            prune_result = client.api.prune_builds()
            result["build_cache_deleted"] = len(prune_result.get("CachesDeleted") or [])
            result["space_reclaimed"] += prune_result.get("SpaceReclaimed", 0)
        except Exception as e:
            logger.warning(f"Failed to prune build cache: {e}")

    return result


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/disk-usage", response_model=list[DiskUsageResponse])
async def get_disk_usage(
    worker_id: int | None = Query(None, description="Filter by worker ID"),
    db: AsyncSession = Depends(get_db),
):
    """Get Docker disk usage for all workers or a specific worker."""
    if worker_id:
        workers = [await get_worker(db, worker_id)]
    else:
        result = await db.execute(select(Worker).where(Worker.status == "online"))
        workers = result.scalars().all()

    if not workers:
        return []

    results = []
    for worker in workers:
        try:
            if _is_local_worker(worker):
                logger.info(f"Getting disk usage from local worker {worker.name}")
                data = _get_local_disk_usage()
            else:
                data = await call_worker(worker, "GET", "/storage/disk-usage")

            results.append(
                DiskUsageResponse(
                    worker_id=worker.id,
                    worker_name=worker.name,
                    images=StorageCategory(**data["images"]),
                    containers=StorageCategory(**data["containers"]),
                    volumes=StorageCategory(**data["volumes"]),
                    build_cache=StorageCategory(**data["build_cache"]),
                    total_size=data["total_size"],
                    total_reclaimable=data["total_reclaimable"],
                )
            )
        except HTTPException as e:
            logger.warning(f"Failed to get disk usage from {worker.name}: {e.detail}")
        except Exception as e:
            logger.warning(f"Failed to get disk usage from {worker.name}: {e}")

    return results


@router.get("/volumes", response_model=list[VolumeResponse])
async def list_volumes(
    worker_id: int | None = Query(None, description="Filter by worker ID"),
    db: AsyncSession = Depends(get_db),
):
    """List Docker volumes across workers."""
    if worker_id:
        workers = [await get_worker(db, worker_id)]
    else:
        result = await db.execute(select(Worker).where(Worker.status == "online"))
        workers = result.scalars().all()

    if not workers:
        return []

    results = []
    for worker in workers:
        try:
            if _is_local_worker(worker):
                logger.info(f"Listing volumes from local worker {worker.name}")
                vols = _get_local_volumes()
                logger.info(f"Local worker {worker.name} has {len(vols)} volumes")
            else:
                logger.info(
                    f"Fetching volumes from worker {worker.name} at {worker.address}"
                )
                data = await call_worker(worker, "GET", "/storage/volumes")
                vols = data.get("items", [])
                logger.info(f"Worker {worker.name} returned {len(vols)} volumes")

            for vol in vols:
                results.append(
                    VolumeResponse(
                        name=vol["name"],
                        worker_id=worker.id,
                        worker_name=worker.name,
                        driver=vol.get("driver", "local"),
                        mountpoint=vol.get("mountpoint", ""),
                        created_at=vol.get("created_at", ""),
                        labels=vol.get("labels") or {},
                        scope=vol.get("scope", "local"),
                    )
                )
        except HTTPException as e:
            logger.warning(f"Failed to list volumes from {worker.name}: {e.detail}")
        except Exception as e:
            logger.warning(f"Failed to list volumes from {worker.name}: {e}")

    return results


@router.delete("/volumes/{volume_name}")
async def delete_volume(
    volume_name: str,
    worker_id: int = Query(..., description="Worker ID"),
    force: bool = Query(False, description="Force delete"),
    db: AsyncSession = Depends(get_db),
):
    """Delete a Docker volume."""
    worker = await get_worker(db, worker_id)

    if _is_local_worker(worker):
        try:
            return _delete_local_volume(volume_name, force)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await call_worker(
        worker,
        "DELETE",
        f"/storage/volumes/{volume_name}",
        params={"force": force},
    )


@router.post("/prune", response_model=list[PruneResponse])
async def prune_storage(
    request: PruneRequest,
    worker_id: int | None = Query(None, description="Filter by worker ID"),
    db: AsyncSession = Depends(get_db),
):
    """Prune unused Docker resources."""
    if worker_id:
        workers = [await get_worker(db, worker_id)]
    else:
        result = await db.execute(select(Worker).where(Worker.status == "online"))
        workers = result.scalars().all()

    if not workers:
        return []

    results = []
    for worker in workers:
        try:
            if _is_local_worker(worker):
                logger.info(f"Pruning storage on local worker {worker.name}")
                data = _prune_local_storage(
                    request.images,
                    request.containers,
                    request.volumes,
                    request.build_cache,
                )
            else:
                data = await call_worker(
                    worker,
                    "POST",
                    "/storage/prune",
                    params={
                        "images": request.images,
                        "containers": request.containers,
                        "volumes": request.volumes,
                        "build_cache": request.build_cache,
                    },
                    timeout=120.0,
                )

            results.append(
                PruneResponse(
                    worker_id=worker.id,
                    worker_name=worker.name,
                    images_deleted=data.get("images_deleted", 0),
                    containers_deleted=data.get("containers_deleted", 0),
                    volumes_deleted=data.get("volumes_deleted", 0),
                    build_cache_deleted=data.get("build_cache_deleted", 0),
                    space_reclaimed=data.get("space_reclaimed", 0),
                )
            )
        except HTTPException as e:
            logger.warning(f"Failed to prune storage on {worker.name}: {e.detail}")
        except Exception as e:
            logger.warning(f"Failed to prune storage on {worker.name}: {e}")

    return results

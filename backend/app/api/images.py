"""Container Image API routes

Provides endpoints for managing Docker images across workers.
All operations are proxied to the appropriate worker agent.
"""

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.worker import Worker

logger = logging.getLogger(__name__)
router = APIRouter()

# HTTP client timeout for image operations (can be slow for large images)
IMAGE_OPERATION_TIMEOUT = 600.0  # 10 minutes
DEFAULT_TIMEOUT = 30.0


# =============================================================================
# Request/Response Schemas
# =============================================================================


class RegistryAuth(BaseModel):
    """Registry authentication credentials."""

    username: str
    password: str
    server_address: str | None = None


class ImagePullRequest(BaseModel):
    """Request to pull an image to a worker."""

    worker_id: int
    image: str = Field(..., description="Image reference (e.g., 'nginx:latest')")
    registry_auth: RegistryAuth | None = None


class ImageBuildRequest(BaseModel):
    """Request to build an image on a worker."""

    worker_id: int
    dockerfile: str = Field(..., description="Dockerfile content")
    tag: str = Field(..., description="Tag for the built image")
    build_args: dict[str, str] | None = None


class ImageResponse(BaseModel):
    """Image information response."""

    id: str
    worker_id: int
    worker_name: str
    repository: str
    tag: str
    full_name: str
    size: int
    created_at: str
    digest: str | None = None
    labels: dict[str, Any] | None = None


class ImageDetailResponse(ImageResponse):
    """Detailed image information including layers."""

    layers: list[dict[str, Any]] = []
    config: dict[str, Any] | None = None


class ImageListResponse(BaseModel):
    """Paginated image list response."""

    items: list[ImageResponse]
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
                raise HTTPException(status_code=404, detail="Resource not found on worker")

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


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("", response_model=ImageListResponse)
async def list_images(
    worker_id: int | None = Query(None, description="Filter by worker ID"),
    repository: str | None = Query(None, description="Filter by repository name"),
    db: AsyncSession = Depends(get_db),
):
    """List images across workers.

    If worker_id is provided, lists images only from that worker.
    Otherwise, lists images from all online workers.
    """
    all_images: list[ImageResponse] = []

    if worker_id:
        # Query specific worker
        workers = [await get_worker_or_404(worker_id, db)]
    else:
        # Query all online workers
        result = await db.execute(select(Worker).where(Worker.status == "online"))
        workers = list(result.scalars().all())

    for worker in workers:
        try:
            params = {}
            if repository:
                params["repository"] = repository

            data = await call_worker_api(worker, "GET", "/images", params=params)

            for img in data.get("items", []):
                all_images.append(
                    ImageResponse(
                        id=img["id"],
                        worker_id=worker.id,
                        worker_name=worker.name,
                        repository=img["repository"],
                        tag=img["tag"],
                        full_name=img["full_name"],
                        size=img["size"],
                        created_at=img["created_at"],
                        digest=img.get("digest"),
                        labels=img.get("labels"),
                    )
                )
        except HTTPException as e:
            if e.status_code != 503:  # Ignore connection errors for listing
                logger.warning(f"Error listing images from {worker.name}: {e.detail}")
        except Exception as e:
            logger.warning(f"Error listing images from {worker.name}: {e}")

    return ImageListResponse(items=all_images, total=len(all_images))


@router.get("/{image_id}", response_model=ImageDetailResponse)
async def get_image(
    image_id: str,
    worker_id: int = Query(..., description="Worker ID where the image is located"),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed information about an image."""
    worker = await get_worker_or_404(worker_id, db)
    data = await call_worker_api(worker, "GET", f"/images/{image_id}")

    return ImageDetailResponse(
        id=data["id"],
        worker_id=worker.id,
        worker_name=worker.name,
        repository=data["repository"],
        tag=data["tag"],
        full_name=data["full_name"],
        size=data["size"],
        created_at=data["created_at"],
        digest=data.get("digest"),
        labels=data.get("labels"),
        layers=data.get("layers", []),
        config=data.get("config"),
    )


@router.post("/pull", status_code=202)
async def pull_image(
    request: ImagePullRequest,
    db: AsyncSession = Depends(get_db),
):
    """Pull an image from a registry to a worker.

    This operation may take a long time for large images.
    Returns immediately with a task ID for progress tracking.
    """
    worker = await get_worker_or_404(request.worker_id, db)

    payload = {
        "image": request.image,
    }
    if request.registry_auth:
        payload["registry_auth"] = request.registry_auth.model_dump()

    data = await call_worker_api(
        worker,
        "POST",
        "/images/pull",
        json=payload,
        timeout=IMAGE_OPERATION_TIMEOUT,
    )

    return {
        "status": "completed",
        "image": data,
        "worker_id": worker.id,
        "worker_name": worker.name,
    }


@router.post("/build", status_code=202)
async def build_image(
    request: ImageBuildRequest,
    db: AsyncSession = Depends(get_db),
):
    """Build an image from a Dockerfile on a worker.

    This operation may take a long time depending on the Dockerfile.
    """
    worker = await get_worker_or_404(request.worker_id, db)

    payload = {
        "dockerfile": request.dockerfile,
        "tag": request.tag,
    }
    if request.build_args:
        payload["build_args"] = request.build_args

    data = await call_worker_api(
        worker,
        "POST",
        "/images/build",
        json=payload,
        timeout=IMAGE_OPERATION_TIMEOUT,
    )

    return {
        "status": "completed",
        "image": data,
        "worker_id": worker.id,
        "worker_name": worker.name,
    }


@router.delete("/{image_id}", status_code=204)
async def delete_image(
    image_id: str,
    worker_id: int = Query(..., description="Worker ID where the image is located"),
    force: bool = Query(False, description="Force removal even if in use"),
    db: AsyncSession = Depends(get_db),
):
    """Delete an image from a worker."""
    worker = await get_worker_or_404(worker_id, db)

    await call_worker_api(
        worker,
        "DELETE",
        f"/images/{image_id}",
        params={"force": force},
    )

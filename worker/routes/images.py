"""Image management routes for LMStack Worker.

Contains endpoints for listing, pulling, building, and removing Docker images.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

try:
    from models import ImageBuildRequest, ImagePullRequest
except ImportError:
    from worker.models import ImageBuildRequest, ImagePullRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])

# Agent reference - set by main app
_agent = None

# =============================================================================
# Progress Tracking for App Image Pulls
# =============================================================================

# In-memory store for app image pull progress
_app_pull_progress: dict[int, dict] = {}
_MAX_PROGRESS_ENTRIES = 100


def get_app_pull_progress(app_id: int) -> dict:
    """Get pull progress for an app."""
    return _app_pull_progress.get(app_id, {"status": "unknown", "progress": 0})


def _set_app_pull_progress(app_id: int, data: dict) -> None:
    """Set pull progress for an app."""
    _app_pull_progress[app_id] = data


def _cleanup_old_app_progress() -> None:
    """Remove old progress entries to prevent memory leaks."""
    if len(_app_pull_progress) > _MAX_PROGRESS_ENTRIES:
        completed_keys = [
            key
            for key, val in _app_pull_progress.items()
            if val.get("status") in ("completed", "error")
        ]
        for key in completed_keys[: len(_app_pull_progress) - _MAX_PROGRESS_ENTRIES // 2]:
            _app_pull_progress.pop(key, None)


def set_agent(agent):
    """Set the agent reference for route handlers."""
    global _agent
    _agent = agent


def get_agent():
    """Get the agent reference or raise error."""
    if _agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    return _agent


@router.get("")
async def list_images(repository: Optional[str] = None):
    """List all images on this worker."""
    agent = get_agent()

    try:
        images = agent.image_manager.list_images(repository=repository)
        return {"items": images, "total": len(images)}
    except Exception as e:
        logger.error(f"Failed to list images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{image_id}")
async def get_image(image_id: str):
    """Get image details."""
    agent = get_agent()

    try:
        return agent.image_manager.get_image_detail(image_id)
    except Exception as e:
        logger.error(f"Failed to get image {image_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")


@router.get("/pull-progress/{app_id}")
async def pull_progress(app_id: int):
    """Get image pull progress for an app."""
    return get_app_pull_progress(app_id)


def _pull_image_with_tracking(agent, image: str, app_id: int, auth_config: dict | None = None):
    """Pull image with progress tracking (runs in background thread)."""
    _cleanup_old_app_progress()

    logger.info(f"Starting image pull for app {app_id}: {image}")

    _set_app_pull_progress(
        app_id,
        {
            "status": "pulling",
            "image": image,
            "progress": 0,
            "layers": {},
        },
    )

    last_logged_progress = 0

    def progress_callback(progress: int, layers: dict):
        """Update progress during pull."""
        nonlocal last_logged_progress
        _set_app_pull_progress(
            app_id,
            {
                "status": "pulling",
                "image": image,
                "progress": progress,
                "layers": layers,
            },
        )
        # Log progress every 10%
        if progress >= last_logged_progress + 10:
            logger.info(f"Pulling image {image} for app {app_id}: {progress}%")
            last_logged_progress = progress

    try:
        result = agent.image_manager.pull_image(
            image=image,
            auth_config=auth_config,
            progress_callback=progress_callback,
        )
        logger.info(f"Image pull completed for app {app_id}: {image}")
        _set_app_pull_progress(
            app_id,
            {
                "status": "completed",
                "image": image,
                "progress": 100,
            },
        )
        return result
    except Exception as e:
        logger.error(f"Image pull failed for app {app_id}: {image} - {e}")
        _set_app_pull_progress(
            app_id,
            {
                "status": "error",
                "image": image,
                "progress": 0,
                "error": str(e),
            },
        )
        raise


@router.post("/pull")
async def pull_image(request: ImagePullRequest):
    """Pull an image from registry."""
    agent = get_agent()

    logger.info(
        f"Received pull request for image: {request.image}"
        + (f" (app_id: {request.app_id})" if request.app_id else "")
    )

    try:
        if request.app_id:
            # Pull with progress tracking in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                _pull_image_with_tracking,
                agent,
                request.image,
                request.app_id,
                request.registry_auth,
            )
            return result
        else:
            # Simple pull without tracking
            logger.info(f"Pulling image without tracking: {request.image}")
            result = agent.image_manager.pull_image(
                image=request.image,
                auth_config=request.registry_auth,
            )
            logger.info(f"Image pull completed: {request.image}")
            return result
    except Exception as e:
        logger.error(f"Failed to pull image {request.image}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/build")
async def build_image(request: ImageBuildRequest):
    """Build an image from Dockerfile."""
    agent = get_agent()

    try:
        result = agent.image_manager.build_image(
            dockerfile=request.dockerfile,
            tag=request.tag,
            build_args=request.build_args,
        )
        return result
    except Exception as e:
        logger.error(f"Failed to build image {request.tag}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{image_id}")
async def delete_image(image_id: str, force: bool = False):
    """Delete an image."""
    agent = get_agent()

    try:
        agent.image_manager.remove_image(image_id, force=force)
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Failed to delete image {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

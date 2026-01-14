"""Image management routes for LMStack Worker.

Contains endpoints for listing, pulling, building, and removing Docker images.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

try:
    from models import ImagePullRequest, ImageBuildRequest
except ImportError:
    from worker.models import ImagePullRequest, ImageBuildRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])

# Agent reference - set by main app
_agent = None


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


@router.post("/pull")
async def pull_image(request: ImagePullRequest):
    """Pull an image from registry."""
    agent = get_agent()

    try:
        result = agent.image_manager.pull_image(
            image=request.image,
            auth_config=request.registry_auth,
        )
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

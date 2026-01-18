"""Semantic Router API endpoints.

Provides endpoints for:
- Checking if Semantic Router is deployed
- Updating Semantic Router config
- Getting Semantic Router status
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.apps.deployment import update_semantic_router_config_if_deployed
from app.database import get_db
from app.services.semantic_router import semantic_router_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/semantic-router", tags=["semantic-router"])


class SemanticRouterStatus(BaseModel):
    """Semantic Router deployment status."""

    deployed: bool
    url: str | None = None
    dashboard_url: str | None = None
    message: str | None = None


class ConfigUpdateResponse(BaseModel):
    """Response for config update."""

    success: bool
    message: str


@router.get("/status", response_model=SemanticRouterStatus)
async def get_semantic_router_status(
    db: AsyncSession = Depends(get_db),
):
    """Check if Semantic Router is deployed and get its URLs."""
    app = await semantic_router_service.get_semantic_router_app(db)

    if not app:
        return SemanticRouterStatus(
            deployed=False,
            message="Semantic Router is not deployed. Deploy it from the Apps page to enable intelligent model routing.",
        )

    if not app.worker:
        return SemanticRouterStatus(
            deployed=False,
            message="Semantic Router worker not found.",
        )

    # Build URLs
    worker_host = app.worker.address.split(":")[0]
    api_url = f"http://{worker_host}:{app.port}"
    dashboard_url = f"http://{worker_host}:{app.port + 1}"  # Dashboard is on port + 1

    return SemanticRouterStatus(
        deployed=True,
        url=api_url,
        dashboard_url=dashboard_url,
        message="Semantic Router is running. Use model='MoM' for automatic routing.",
    )


@router.post("/update-config", response_model=ConfigUpdateResponse)
async def update_semantic_router_config(
    db: AsyncSession = Depends(get_db),
):
    """Update Semantic Router config with latest deployments.

    This endpoint regenerates the config.yaml with current running models
    and writes it to the Semantic Router volume. The router will automatically
    reload the config (hot-reload supported).
    """
    try:
        updated = await update_semantic_router_config_if_deployed(db)

        if updated:
            return ConfigUpdateResponse(
                success=True,
                message="Semantic Router config updated successfully. Changes will take effect automatically.",
            )
        else:
            return ConfigUpdateResponse(
                success=False,
                message="Semantic Router is not deployed. Deploy it first from the Apps page.",
            )

    except Exception as e:
        logger.error(f"Failed to update semantic router config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config-preview")
async def preview_semantic_router_config(
    db: AsyncSession = Depends(get_db),
):
    """Preview the Semantic Router config that would be generated.

    This is useful for debugging or understanding how the config is built.
    """
    lmstack_api_url = "http://host.docker.internal:52000"
    config = await semantic_router_service.generate_config(db, lmstack_api_url)
    return config

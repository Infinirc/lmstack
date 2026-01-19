"""Semantic Router API endpoints.

Provides endpoints for:
- Checking if Semantic Router is deployed
- Updating Semantic Router config
- Getting Semantic Router status
- Proxying chat requests to Semantic Router (for logged-in users)
"""

import json
import logging
from collections.abc import AsyncGenerator

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.apps.deployment import update_semantic_router_config_if_deployed
from app.core.deps import require_viewer
from app.database import get_db
from app.models.user import User
from app.services.semantic_router import semantic_router_service

logger = logging.getLogger(__name__)

# Timeout for chat requests (5 minutes for long model responses)
CHAT_PROXY_TIMEOUT = 300.0

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
    # Try to get lmstack_host from deployed Semantic Router config, fallback to placeholder
    app = await semantic_router_service.get_semantic_router_app(db)
    if app:
        app_config = app.config or {}
        lmstack_host = app_config.get("lmstack_host")
        if lmstack_host:
            lmstack_api_url = f"http://{lmstack_host}:52000"
        elif app.worker:
            # Fallback to worker IP (may not be correct for container access)
            worker_host = app.worker.address.split(":")[0]
            lmstack_api_url = f"http://{worker_host}:52000"
        else:
            lmstack_api_url = "http://<lmstack-host>:52000"
    else:
        lmstack_api_url = "http://<lmstack-host>:52000"
    config = await semantic_router_service.generate_config(db, lmstack_api_url)
    return config


@router.post("/chat")
async def proxy_semantic_router_chat(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Proxy chat requests to Semantic Router (requires viewer+).

    This endpoint allows logged-in users to chat with the Semantic Router
    using JWT authentication instead of API keys. The Semantic Router will
    intelligently select the best model for each request.
    """
    # Check if Semantic Router is deployed
    router_app = await semantic_router_service.get_semantic_router_app(db)
    if not router_app or router_app.status != "running":
        raise HTTPException(
            status_code=503,
            detail="Semantic Router is not deployed or not running. Deploy it from the Apps page.",
        )

    router_url = await semantic_router_service.get_semantic_router_url(db)
    if not router_url:
        raise HTTPException(status_code=503, detail="Semantic Router URL not available")

    # Get request body
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Get the default model from running deployments
    # Semantic Router requires a model field, it will route based on domain detection
    model_name = body.get("model", "")
    if not model_name or model_name.lower() in ("mom", "mom (intelligent router)"):
        # Query the first running deployment to get a valid model name
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        from app.models.deployment import Deployment, DeploymentStatus

        result = await db.execute(
            select(Deployment)
            .where(Deployment.status == DeploymentStatus.RUNNING.value)
            .options(selectinload(Deployment.model))
            .limit(1)
        )
        deployment = result.scalar_one_or_none()
        if deployment and deployment.model:
            # Use the model name format that matches Semantic Router config
            default_model = deployment.model.name.replace("/", "-").replace(":", "-")
        else:
            default_model = "default"
        body["model"] = default_model

    # Get API key from app config for Semantic Router authentication
    app_config = router_app.config or {}
    api_key = app_config.get("api_key")

    # Check if streaming
    is_streaming = body.get("stream", False)

    chat_endpoint = f"{router_url}/v1/chat/completions"

    if is_streaming:
        return await _proxy_streaming_chat(chat_endpoint, body, api_key)
    else:
        return await _proxy_chat(chat_endpoint, body, api_key)


async def _proxy_chat(upstream_url: str, body: dict, api_key: str | None = None) -> dict:
    """Proxy a non-streaming chat request."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=CHAT_PROXY_TIMEOUT) as client:
            response = await client.post(
                upstream_url,
                json=body,
                headers=headers,
            )
            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Semantic Router timed out")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Failed to connect to Semantic Router")
    except httpx.RequestError as e:
        logger.error(f"Chat proxy request error: {e}")
        raise HTTPException(status_code=502, detail=f"Request error: {str(e)}")


async def _proxy_streaming_chat(
    upstream_url: str, body: dict, api_key: str | None = None
) -> StreamingResponse:
    """Proxy a streaming chat request."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def stream_generator() -> AsyncGenerator[bytes, None]:
        try:
            timeout = httpx.Timeout(CHAT_PROXY_TIMEOUT, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    upstream_url,
                    json=body,
                    headers=headers,
                ) as response:
                    # Stream each line separately for better real-time delivery
                    async for line in response.aiter_lines():
                        if line:
                            yield (line + "\n").encode()

        except httpx.TimeoutException:
            logger.error(f"Streaming timeout for {upstream_url}")
            error_data = {
                "error": {
                    "message": "Request to Semantic Router timed out",
                    "type": "timeout_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n".encode()
        except httpx.ConnectError:
            logger.error(f"Connection error for {upstream_url}")
            error_data = {
                "error": {
                    "message": "Failed to connect to Semantic Router",
                    "type": "connection_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n".encode()
        except httpx.RequestError as e:
            logger.error(f"Streaming request error: {e}")
            error_data = {"error": {"message": f"Request error: {str(e)}", "type": "request_error"}}
            yield f"data: {json.dumps(error_data)}\n\n".encode()

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

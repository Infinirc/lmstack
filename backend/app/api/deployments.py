"""Deployment API routes"""

import json
import logging
from collections.abc import AsyncGenerator

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.deps import require_operator, require_viewer
from app.database import get_db
from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import LLMModel
from app.models.user import User
from app.models.worker import Worker
from app.schemas.deployment import (
    DeploymentCreate,
    DeploymentListResponse,
    DeploymentLogsResponse,
    DeploymentResponse,
    DeploymentUpdate,
    ModelSummary,
    WorkerSummary,
)
from app.services.deployer import DeployerService
from app.services.gateway import gateway_service

logger = logging.getLogger(__name__)

router = APIRouter()


def deployment_to_response(deployment: Deployment) -> DeploymentResponse:
    """Convert deployment model to response schema"""
    worker_summary = None
    if deployment.worker:
        worker_summary = WorkerSummary(
            id=deployment.worker.id,
            name=deployment.worker.name,
            address=deployment.worker.address,
            status=deployment.worker.status,
        )

    model_summary = None
    if deployment.model:
        model_summary = ModelSummary(
            id=deployment.model.id,
            name=deployment.model.name,
            model_id=deployment.model.model_id,
            source=deployment.model.source,
        )

    return DeploymentResponse(
        id=deployment.id,
        name=deployment.name,
        model_id=deployment.model_id,
        worker_id=deployment.worker_id,
        backend=deployment.backend,
        status=deployment.status,
        status_message=deployment.status_message,
        container_id=deployment.container_id,
        port=deployment.port,
        gpu_indexes=deployment.gpu_indexes,
        extra_params=deployment.extra_params,
        created_at=deployment.created_at,
        updated_at=deployment.updated_at,
        worker=worker_summary,
        model=model_summary,
    )


@router.get("", response_model=DeploymentListResponse)
async def list_deployments(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: str | None = None,
    worker_id: int | None = None,
    model_id: int | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """List all deployments (requires viewer+)"""
    query = select(Deployment).options(
        selectinload(Deployment.worker),
        selectinload(Deployment.model),
    )

    if status:
        query = query.where(Deployment.status == status)
    if worker_id:
        query = query.where(Deployment.worker_id == worker_id)
    if model_id:
        query = query.where(Deployment.model_id == model_id)

    # Get total count
    count_query = select(func.count()).select_from(
        select(Deployment)
        .where(
            *([Deployment.status == status] if status else []),
            *([Deployment.worker_id == worker_id] if worker_id else []),
            *([Deployment.model_id == model_id] if model_id else []),
        )
        .subquery()
    )
    total = await db.scalar(count_query)

    # Get paginated results
    query = query.offset(skip).limit(limit).order_by(Deployment.created_at.desc())
    result = await db.execute(query)
    deployments = result.scalars().all()

    items = [deployment_to_response(d) for d in deployments]

    return DeploymentListResponse(items=items, total=total or 0)


@router.post("", response_model=DeploymentResponse, status_code=201)
async def create_deployment(
    deployment_in: DeploymentCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Create a new deployment (requires operator+)"""
    # Check if deployment with same name exists
    existing = await db.execute(select(Deployment).where(Deployment.name == deployment_in.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Deployment with this name already exists")

    # Verify worker exists
    worker_result = await db.execute(select(Worker).where(Worker.id == deployment_in.worker_id))
    worker = worker_result.scalar_one_or_none()
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Verify model exists
    model_result = await db.execute(select(LLMModel).where(LLMModel.id == deployment_in.model_id))
    model = model_result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Validate backend compatibility with model source
    from app.models.llm_model import BackendType, ModelSource

    backend_value = (
        deployment_in.backend.value
        if hasattr(deployment_in.backend, "value")
        else deployment_in.backend
    )

    if model.source == ModelSource.OLLAMA.value:
        # Ollama models can only use Ollama backend
        if backend_value != BackendType.OLLAMA.value:
            raise HTTPException(
                status_code=400,
                detail="Ollama models can only be deployed with Ollama backend",
            )
    elif model.source == ModelSource.HUGGINGFACE.value:
        # HuggingFace models can use vLLM or SGLang, not Ollama
        if backend_value == BackendType.OLLAMA.value:
            raise HTTPException(
                status_code=400,
                detail="HuggingFace models cannot be deployed with Ollama backend. Use vLLM or SGLang.",
            )

    deployment = Deployment(
        name=deployment_in.name,
        model_id=deployment_in.model_id,
        worker_id=deployment_in.worker_id,
        backend=backend_value,
        gpu_indexes=deployment_in.gpu_indexes,
        extra_params=deployment_in.extra_params,
        status=DeploymentStatus.PENDING.value,
    )

    db.add(deployment)
    await db.commit()
    await db.refresh(deployment)

    # Start deployment in background
    deployer = DeployerService()
    background_tasks.add_task(deployer.deploy, deployment.id)

    # Reload with relationships
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == deployment.id)
        .options(
            selectinload(Deployment.worker),
            selectinload(Deployment.model),
        )
    )
    deployment = result.scalar_one()

    return deployment_to_response(deployment)


@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Get a deployment by ID (requires viewer+)"""
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == deployment_id)
        .options(
            selectinload(Deployment.worker),
            selectinload(Deployment.model),
        )
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    return deployment_to_response(deployment)


@router.patch("/{deployment_id}", response_model=DeploymentResponse)
async def update_deployment(
    deployment_id: int,
    deployment_in: DeploymentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Update a deployment (requires operator+)"""
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == deployment_id)
        .options(
            selectinload(Deployment.worker),
            selectinload(Deployment.model),
        )
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    update_data = deployment_in.model_dump(exclude_unset=True)

    if "status" in update_data and update_data["status"]:
        update_data["status"] = update_data["status"].value

    for field, value in update_data.items():
        setattr(deployment, field, value)

    await db.commit()
    await db.refresh(deployment)

    return deployment_to_response(deployment)


@router.delete("/{deployment_id}", status_code=204)
async def delete_deployment(
    deployment_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Stop and delete a deployment (requires operator+)"""
    result = await db.execute(select(Deployment).where(Deployment.id == deployment_id))
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    # Stop container in background if running
    if deployment.container_id:
        deployer = DeployerService()
        background_tasks.add_task(deployer.stop, deployment.id)

    await db.delete(deployment)
    await db.commit()


@router.post("/{deployment_id}/stop", response_model=DeploymentResponse)
async def stop_deployment(
    deployment_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Stop a deployment without deleting it (requires operator+)"""
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == deployment_id)
        .options(
            selectinload(Deployment.worker),
            selectinload(Deployment.model),
        )
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if deployment.status == DeploymentStatus.STOPPED.value:
        raise HTTPException(status_code=400, detail="Deployment is already stopped")

    # Stop container in background if running
    if deployment.container_id:
        deployer = DeployerService()
        background_tasks.add_task(deployer.stop, deployment.id)

    deployment.status = DeploymentStatus.STOPPED.value
    deployment.status_message = "Stopped by user"
    await db.commit()
    await db.refresh(deployment)

    return deployment_to_response(deployment)


@router.post("/{deployment_id}/start", response_model=DeploymentResponse)
async def start_deployment(
    deployment_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Restart a stopped deployment (requires operator+)"""
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == deployment_id)
        .options(
            selectinload(Deployment.worker),
            selectinload(Deployment.model),
        )
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if deployment.status not in [
        DeploymentStatus.STOPPED.value,
        DeploymentStatus.ERROR.value,
    ]:
        raise HTTPException(status_code=400, detail="Deployment is not stopped or in error state")

    # Reset status and start deployment
    deployment.status = DeploymentStatus.PENDING.value
    deployment.status_message = "Restarting..."
    deployment.container_id = None
    await db.commit()

    # Start deployment in background
    deployer = DeployerService()
    background_tasks.add_task(deployer.deploy, deployment.id)

    await db.refresh(deployment)
    return deployment_to_response(deployment)


@router.get("/{deployment_id}/logs", response_model=DeploymentLogsResponse)
async def get_deployment_logs(
    deployment_id: int,
    tail: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Get logs for a deployment (requires viewer+)"""
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == deployment_id)
        .options(selectinload(Deployment.worker))
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if not deployment.container_id:
        return DeploymentLogsResponse(
            deployment_id=deployment_id,
            logs="No container running",
        )

    deployer = DeployerService()
    logs = await deployer.get_logs(deployment, tail=tail)

    return DeploymentLogsResponse(deployment_id=deployment_id, logs=logs)


# Chat proxy timeout (5 minutes for long model responses)
CHAT_PROXY_TIMEOUT = 300.0


@router.post("/{deployment_id}/chat")
async def proxy_chat(
    deployment_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Proxy chat requests to deployment (requires viewer+).

    This endpoint proxies chat completion requests to the model container,
    allowing the frontend to communicate with models without needing direct
    network access to Docker internal IPs.

    The request body should be an OpenAI-compatible chat completion request.
    Supports both streaming and non-streaming responses.
    """
    # Get deployment with worker info
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == deployment_id)
        .options(
            selectinload(Deployment.worker),
            selectinload(Deployment.model),
        )
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if deployment.status != DeploymentStatus.RUNNING.value:
        raise HTTPException(
            status_code=400,
            detail=f"Deployment is not running (status: {deployment.status})",
        )

    if not deployment.worker or not deployment.port:
        raise HTTPException(status_code=400, detail="Deployment has no worker or port assigned")

    # Build upstream URL using the gateway service (handles Docker networking correctly)
    upstream_url = gateway_service.build_upstream_url(
        deployment.worker.address,
        deployment.port,
        deployment.container_name,
    )
    chat_endpoint = f"{upstream_url}/v1/chat/completions"

    # Get request body
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Check if streaming
    is_streaming = body.get("stream", False)

    logger.debug(f"Proxying chat to {chat_endpoint}, streaming={is_streaming}")

    if is_streaming:
        return await _proxy_streaming_chat(chat_endpoint, body)
    else:
        return await _proxy_chat(chat_endpoint, body)


async def _proxy_chat(upstream_url: str, body: dict) -> dict:
    """Proxy a non-streaming chat request."""
    try:
        async with httpx.AsyncClient(timeout=CHAT_PROXY_TIMEOUT) as client:
            response = await client.post(
                upstream_url,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to model timed out")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Failed to connect to model")
    except httpx.RequestError as e:
        logger.error(f"Chat proxy request error: {e}")
        raise HTTPException(status_code=502, detail=f"Request error: {str(e)}")


async def _proxy_streaming_chat(upstream_url: str, body: dict) -> StreamingResponse:
    """Proxy a streaming chat request."""

    async def stream_generator() -> AsyncGenerator[bytes, None]:
        try:
            async with httpx.AsyncClient(timeout=CHAT_PROXY_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    upstream_url,
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        except httpx.TimeoutException:
            logger.error(f"Streaming timeout for {upstream_url}")
            error_data = {
                "error": {"message": "Request to model timed out", "type": "timeout_error"}
            }
            yield f"data: {json.dumps(error_data)}\n\n".encode()
        except httpx.ConnectError:
            logger.error(f"Connection error for {upstream_url}")
            error_data = {
                "error": {"message": "Failed to connect to model", "type": "connection_error"}
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

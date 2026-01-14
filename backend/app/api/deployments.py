"""Deployment API routes"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import LLMModel
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
):
    """List all deployments"""
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
):
    """Create a new deployment"""
    # Check if deployment with same name exists
    existing = await db.execute(
        select(Deployment).where(Deployment.name == deployment_in.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=400, detail="Deployment with this name already exists"
        )

    # Verify worker exists
    worker_result = await db.execute(
        select(Worker).where(Worker.id == deployment_in.worker_id)
    )
    worker = worker_result.scalar_one_or_none()
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Verify model exists
    model_result = await db.execute(
        select(LLMModel).where(LLMModel.id == deployment_in.model_id)
    )
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
):
    """Get a deployment by ID"""
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
):
    """Update a deployment"""
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
):
    """Stop and delete a deployment"""
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
):
    """Stop a deployment without deleting it"""
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
):
    """Restart a stopped deployment"""
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
        raise HTTPException(
            status_code=400, detail="Deployment is not stopped or in error state"
        )

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
):
    """Get logs for a deployment"""
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

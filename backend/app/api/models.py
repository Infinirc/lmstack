"""LLM Model API routes"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import require_operator, require_viewer
from app.database import get_db
from app.models.deployment import Deployment
from app.models.llm_model import LLMModel
from app.models.user import User
from app.schemas.llm_model import (
    LLMModelCreate,
    LLMModelListResponse,
    LLMModelResponse,
    LLMModelUpdate,
)

router = APIRouter()


@router.get("", response_model=LLMModelListResponse)
async def list_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    source: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """List all LLM models (requires viewer+)"""
    query = select(LLMModel)

    if source:
        query = query.where(LLMModel.source == source)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Get paginated results
    query = query.offset(skip).limit(limit).order_by(LLMModel.created_at.desc())
    result = await db.execute(query)
    models = result.scalars().all()

    # Add deployment count
    items = []
    for model in models:
        deployment_count_query = select(func.count()).where(Deployment.model_id == model.id)
        deployment_count = await db.scalar(deployment_count_query) or 0

        model_dict = {
            "id": model.id,
            "name": model.name,
            "model_id": model.model_id,
            "source": model.source,
            "description": model.description,
            "default_params": model.default_params,
            "docker_image": model.docker_image,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "deployment_count": deployment_count,
        }
        items.append(LLMModelResponse(**model_dict))

    return LLMModelListResponse(items=items, total=total or 0)


@router.post("", response_model=LLMModelResponse, status_code=201)
async def create_model(
    model_in: LLMModelCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Create a new LLM model definition (requires operator+)"""
    # Check if model with same name exists
    existing = await db.execute(select(LLMModel).where(LLMModel.name == model_in.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Model with this name already exists")

    source_value = model_in.source.value if model_in.source else "huggingface"
    # Set default backend based on source for backwards compatibility
    default_backend = "ollama" if source_value == "ollama" else "vllm"

    model = LLMModel(
        name=model_in.name,
        model_id=model_in.model_id,
        source=source_value,
        backend=default_backend,  # Keep for backwards compatibility
        description=model_in.description,
        default_params=model_in.default_params,
        docker_image=model_in.docker_image,
    )

    db.add(model)
    await db.commit()
    await db.refresh(model)

    return LLMModelResponse(
        id=model.id,
        name=model.name,
        model_id=model.model_id,
        source=model.source,
        description=model.description,
        default_params=model.default_params,
        docker_image=model.docker_image,
        created_at=model.created_at,
        updated_at=model.updated_at,
        deployment_count=0,
    )


@router.get("/{model_id}", response_model=LLMModelResponse)
async def get_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Get an LLM model by ID (requires viewer+)"""
    result = await db.execute(select(LLMModel).where(LLMModel.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    deployment_count_query = select(func.count()).where(Deployment.model_id == model.id)
    deployment_count = await db.scalar(deployment_count_query) or 0

    return LLMModelResponse(
        id=model.id,
        name=model.name,
        model_id=model.model_id,
        source=model.source,
        description=model.description,
        default_params=model.default_params,
        docker_image=model.docker_image,
        created_at=model.created_at,
        updated_at=model.updated_at,
        deployment_count=deployment_count,
    )


@router.patch("/{model_id}", response_model=LLMModelResponse)
async def update_model(
    model_id: int,
    model_in: LLMModelUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Update an LLM model (requires operator+)"""
    result = await db.execute(select(LLMModel).where(LLMModel.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    update_data = model_in.model_dump(exclude_unset=True)

    if "source" in update_data and update_data["source"]:
        update_data["source"] = update_data["source"].value

    for field, value in update_data.items():
        setattr(model, field, value)

    await db.commit()
    await db.refresh(model)

    deployment_count_query = select(func.count()).where(Deployment.model_id == model.id)
    deployment_count = await db.scalar(deployment_count_query) or 0

    return LLMModelResponse(
        id=model.id,
        name=model.name,
        model_id=model.model_id,
        source=model.source,
        description=model.description,
        default_params=model.default_params,
        docker_image=model.docker_image,
        created_at=model.created_at,
        updated_at=model.updated_at,
        deployment_count=deployment_count,
    )


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Delete an LLM model (requires operator+)"""
    result = await db.execute(select(LLMModel).where(LLMModel.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if model has active deployments
    deployment_count = await db.scalar(select(func.count()).where(Deployment.model_id == model_id))
    if deployment_count and deployment_count > 0:
        raise HTTPException(status_code=400, detail="Cannot delete model with active deployments")

    await db.delete(model)
    await db.commit()

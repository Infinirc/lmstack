"""API Keys management routes"""

import hashlib
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.schemas.api_key import (
    ApiKeyCreate,
    ApiKeyCreateResponse,
    ApiKeyListResponse,
    ApiKeyResponse,
    ApiKeyUpdate,
)

router = APIRouter()

API_KEY_PREFIX = "lmsk"


def generate_access_key() -> str:
    """Generate a random access key"""
    return secrets.token_hex(8)  # 16 characters


def generate_secret_key() -> str:
    """Generate a random secret key"""
    return secrets.token_hex(16)  # 32 characters


def hash_secret(secret: str) -> str:
    """Hash a secret key for storage"""
    return hashlib.sha256(secret.encode()).hexdigest()


def verify_secret(secret: str, hashed: str) -> bool:
    """Verify a secret against its hash"""
    return hash_secret(secret) == hashed


@router.get("", response_model=ApiKeyListResponse)
async def list_api_keys(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys"""
    # Count total
    total = await db.scalar(select(func.count()).select_from(ApiKey))

    # Get paginated results
    result = await db.execute(
        select(ApiKey).offset(skip).limit(limit).order_by(ApiKey.created_at.desc())
    )
    api_keys = result.scalars().all()

    return ApiKeyListResponse(
        items=[ApiKeyResponse.model_validate(k) for k in api_keys],
        total=total or 0,
    )


@router.post("", response_model=ApiKeyCreateResponse, status_code=201)
async def create_api_key(
    api_key_in: ApiKeyCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key"""
    # Check for duplicate name
    existing = await db.execute(select(ApiKey).where(ApiKey.name == api_key_in.name))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=400, detail="API key with this name already exists"
        )

    # Generate keys
    access_key = generate_access_key()
    secret_key = generate_secret_key()
    full_key = f"{API_KEY_PREFIX}_{access_key}_{secret_key}"

    # Calculate expiration
    expires_at = None
    if api_key_in.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=api_key_in.expires_in_days)

    # Create API key record
    api_key = ApiKey(
        name=api_key_in.name,
        description=api_key_in.description,
        access_key=access_key,
        hashed_secret=hash_secret(secret_key),
        allowed_model_ids=api_key_in.allowed_model_ids,
        monthly_token_limit=api_key_in.monthly_token_limit,
        expires_at=expires_at,
    )

    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    # Return with full key (only shown once)
    response = ApiKeyCreateResponse(
        id=api_key.id,
        name=api_key.name,
        description=api_key.description,
        access_key=api_key.access_key,
        allowed_model_ids=api_key.allowed_model_ids,
        monthly_token_limit=api_key.monthly_token_limit,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        api_key=full_key,
    )

    return response


@router.get("/{api_key_id}", response_model=ApiKeyResponse)
async def get_api_key(
    api_key_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get an API key by ID"""
    result = await db.execute(select(ApiKey).where(ApiKey.id == api_key_id))
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    return ApiKeyResponse.model_validate(api_key)


@router.patch("/{api_key_id}", response_model=ApiKeyResponse)
async def update_api_key(
    api_key_id: int,
    api_key_in: ApiKeyUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update an API key"""
    result = await db.execute(select(ApiKey).where(ApiKey.id == api_key_id))
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    # Check for duplicate name
    if api_key_in.name and api_key_in.name != api_key.name:
        existing = await db.execute(
            select(ApiKey).where(ApiKey.name == api_key_in.name)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=400, detail="API key with this name already exists"
            )

    # Update fields
    update_data = api_key_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(api_key, field, value)

    await db.commit()
    await db.refresh(api_key)

    return ApiKeyResponse.model_validate(api_key)


@router.delete("/{api_key_id}", status_code=204)
async def delete_api_key(
    api_key_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete an API key"""
    result = await db.execute(select(ApiKey).where(ApiKey.id == api_key_id))
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    await db.delete(api_key)
    await db.commit()


@router.get("/{api_key_id}/stats")
async def get_api_key_stats(
    api_key_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get usage statistics for an API key"""
    from datetime import timedelta

    from app.models.api_key import Usage

    result = await db.execute(select(ApiKey).where(ApiKey.id == api_key_id))
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    # Get usage stats for last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    usage_result = await db.execute(
        select(
            func.sum(Usage.request_count).label("total_requests"),
            func.sum(Usage.prompt_tokens).label("total_prompt_tokens"),
            func.sum(Usage.completion_tokens).label("total_completion_tokens"),
        )
        .where(Usage.api_key_id == api_key_id)
        .where(Usage.date >= thirty_days_ago)
    )
    row = usage_result.first()

    return {
        "api_key_id": api_key_id,
        "total_requests": row.total_requests or 0 if row else 0,
        "total_prompt_tokens": row.total_prompt_tokens or 0 if row else 0,
        "total_completion_tokens": row.total_completion_tokens or 0 if row else 0,
        "total_tokens": (
            (row.total_prompt_tokens or 0) + (row.total_completion_tokens or 0)
            if row
            else 0
        ),
    }


@router.get("/stats/summary")
async def get_all_api_keys_stats(
    db: AsyncSession = Depends(get_db),
):
    """Get summary statistics for all API keys"""
    from datetime import timedelta

    from app.models.api_key import Usage

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    # Total stats
    total_result = await db.execute(
        select(
            func.sum(Usage.request_count).label("total_requests"),
            func.sum(Usage.prompt_tokens).label("total_prompt_tokens"),
            func.sum(Usage.completion_tokens).label("total_completion_tokens"),
        ).where(Usage.date >= thirty_days_ago)
    )
    total_row = total_result.first()

    # Per API key stats
    per_key_result = await db.execute(
        select(
            Usage.api_key_id,
            func.sum(Usage.request_count).label("requests"),
            func.sum(Usage.prompt_tokens + Usage.completion_tokens).label("tokens"),
        )
        .where(Usage.date >= thirty_days_ago)
        .where(Usage.api_key_id.isnot(None))
        .group_by(Usage.api_key_id)
    )
    per_key_stats = {
        row.api_key_id: {"requests": row.requests or 0, "tokens": row.tokens or 0}
        for row in per_key_result
    }

    return {
        "total_requests": total_row.total_requests or 0 if total_row else 0,
        "total_prompt_tokens": total_row.total_prompt_tokens or 0 if total_row else 0,
        "total_completion_tokens": (
            total_row.total_completion_tokens or 0 if total_row else 0
        ),
        "total_tokens": (
            (total_row.total_prompt_tokens or 0)
            + (total_row.total_completion_tokens or 0)
            if total_row
            else 0
        ),
        "per_key_stats": per_key_stats,
    }

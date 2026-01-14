"""Dashboard API routes"""

from datetime import datetime, timedelta, date
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.worker import Worker
from app.models.llm_model import LLMModel
from app.models.deployment import Deployment, DeploymentStatus
from app.models.api_key import ApiKey, Usage
from app.schemas.dashboard import (
    DashboardResponse,
    ResourceCounts,
    GPUSummary,
    UsageSummary,
    UsagePoint,
    TopModel,
    TopApiKey,
)

router = APIRouter()


@router.get("", response_model=DashboardResponse)
async def get_dashboard(db: AsyncSession = Depends(get_db)):
    """Get dashboard statistics"""

    # Resource counts
    workers = (await db.execute(select(Worker))).scalars().all()
    worker_count = len(workers)
    worker_online_count = sum(1 for w in workers if w.status == "online")

    # GPU count from online workers
    gpu_count = 0
    total_memory = 0
    used_memory = 0
    total_utilization = 0
    gpu_with_util = 0
    total_temperature = 0
    gpu_with_temp = 0
    max_temperature = 0

    for worker in workers:
        if worker.status == "online" and worker.gpu_info:
            for gpu in worker.gpu_info:
                gpu_count += 1
                total_memory += gpu.get("memory_total", 0)
                used_memory += gpu.get("memory_used", 0)
                if gpu.get("utilization") is not None:
                    total_utilization += gpu.get("utilization", 0)
                    gpu_with_util += 1
                # Temperature stats
                temp = gpu.get("temperature", 0)
                if temp and temp > 0:
                    total_temperature += temp
                    gpu_with_temp += 1
                    if temp > max_temperature:
                        max_temperature = temp

    # Model and deployment counts
    model_count = await db.scalar(select(func.count()).select_from(LLMModel))
    deployment_count = await db.scalar(select(func.count()).select_from(Deployment))
    deployment_running_count = await db.scalar(
        select(func.count())
        .select_from(Deployment)
        .where(Deployment.status == DeploymentStatus.RUNNING.value)
    )

    resources = ResourceCounts(
        worker_count=worker_count,
        worker_online_count=worker_online_count,
        gpu_count=gpu_count,
        model_count=model_count or 0,
        deployment_count=deployment_count or 0,
        deployment_running_count=deployment_running_count or 0,
    )

    gpu_summary = GPUSummary(
        total_memory_gb=round(total_memory / (1024**3), 2) if total_memory else 0,
        used_memory_gb=round(used_memory / (1024**3), 2) if used_memory else 0,
        utilization_avg=round(total_utilization / gpu_with_util, 1) if gpu_with_util else 0,
        temperature_avg=round(total_temperature / gpu_with_temp, 1) if gpu_with_temp else 0,
        temperature_max=max_temperature,
    )

    # Usage statistics (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    usage_result = await db.execute(
        select(
            func.date(Usage.date).label("day"),
            func.sum(Usage.request_count).label("requests"),
            func.sum(Usage.prompt_tokens + Usage.completion_tokens).label("tokens"),
        )
        .where(Usage.date >= thirty_days_ago)
        .group_by(func.date(Usage.date))
        .order_by(func.date(Usage.date))
    )
    usage_rows = usage_result.all()

    request_history = [UsagePoint(date=row.day, value=row.requests or 0) for row in usage_rows]
    token_history = [UsagePoint(date=row.day, value=row.tokens or 0) for row in usage_rows]

    # Total usage
    total_result = await db.execute(
        select(
            func.sum(Usage.request_count).label("requests"),
            func.sum(Usage.prompt_tokens).label("prompt"),
            func.sum(Usage.completion_tokens).label("completion"),
        ).where(Usage.date >= thirty_days_ago)
    )
    total_row = total_result.first()

    usage = UsageSummary(
        total_requests=total_row.requests or 0 if total_row else 0,
        total_prompt_tokens=total_row.prompt or 0 if total_row else 0,
        total_completion_tokens=total_row.completion or 0 if total_row else 0,
        request_history=request_history,
        token_history=token_history,
    )

    # Top models by usage
    top_models_result = await db.execute(
        select(
            Usage.model_id,
            LLMModel.name,
            func.sum(Usage.request_count).label("requests"),
            func.sum(Usage.prompt_tokens + Usage.completion_tokens).label("tokens"),
        )
        .join(LLMModel, Usage.model_id == LLMModel.id)
        .where(Usage.date >= thirty_days_ago)
        .group_by(Usage.model_id, LLMModel.name)
        .order_by(func.sum(Usage.request_count).desc())
        .limit(5)
    )
    top_models = [
        TopModel(
            model_id=row.model_id,
            model_name=row.name,
            request_count=row.requests or 0,
            token_count=row.tokens or 0,
        )
        for row in top_models_result
    ]

    # Top API keys by usage
    top_keys_result = await db.execute(
        select(
            Usage.api_key_id,
            ApiKey.name,
            func.sum(Usage.request_count).label("requests"),
            func.sum(Usage.prompt_tokens + Usage.completion_tokens).label("tokens"),
        )
        .join(ApiKey, Usage.api_key_id == ApiKey.id)
        .where(Usage.date >= thirty_days_ago)
        .where(Usage.api_key_id.isnot(None))
        .group_by(Usage.api_key_id, ApiKey.name)
        .order_by(func.sum(Usage.request_count).desc())
        .limit(5)
    )
    top_api_keys = [
        TopApiKey(
            api_key_id=row.api_key_id,
            api_key_name=row.name,
            request_count=row.requests or 0,
            token_count=row.tokens or 0,
        )
        for row in top_keys_result
    ]

    return DashboardResponse(
        resources=resources,
        gpu_summary=gpu_summary,
        usage=usage,
        top_models=top_models,
        top_api_keys=top_api_keys,
    )

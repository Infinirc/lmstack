"""
Model Files API Routes

Provides a virtual view of model files on workers by aggregating deployment data.
A model "exists" on a worker when there's any deployment for it on that worker.
"""

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.deployment import Deployment, DeploymentStatus
from app.schemas.model_files import ModelFileDeployment, ModelFileListResponse, ModelFileView

router = APIRouter()


@router.get("", response_model=ModelFileListResponse)
async def list_model_files(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    worker_id: int | None = None,
    model_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List model files across all workers.

    This endpoint aggregates deployment data to show which models exist on
    which workers. Each unique (model, worker) pair is represented as a
    "model file" entry.

    Filters:
    - worker_id: Show only models on a specific worker
    - model_id: Show only a specific model across all workers
    """
    # Build query with eager loading
    query = select(Deployment).options(
        selectinload(Deployment.worker),
        selectinload(Deployment.model),
    )

    # Apply filters
    if worker_id:
        query = query.where(Deployment.worker_id == worker_id)
    if model_id:
        query = query.where(Deployment.model_id == model_id)

    result = await db.execute(query)
    deployments = result.scalars().all()

    # Group deployments by (model_id, worker_id) pair
    grouped: dict[tuple[int, int], list[Deployment]] = defaultdict(list)
    for d in deployments:
        if d.model_id and d.worker_id:  # Skip orphaned deployments
            key = (d.model_id, d.worker_id)
            grouped[key].append(d)

    # Build model file views
    model_files = []
    for (mid, wid), deps in grouped.items():
        first_dep = deps[0]

        # Determine overall status based on deployment states
        running_count = sum(1 for d in deps if d.status == DeploymentStatus.RUNNING.value)
        downloading_count = sum(1 for d in deps if d.status == DeploymentStatus.DOWNLOADING.value)
        starting_count = sum(
            1
            for d in deps
            if d.status
            in [
                DeploymentStatus.PENDING.value,
                DeploymentStatus.STARTING.value,
            ]
        )

        # Status priority: downloading > starting > running > stopped
        if downloading_count > 0:
            status = "downloading"
        elif starting_count > 0:
            status = "starting"
        elif running_count > 0:
            status = "ready"
        else:
            status = "stopped"

        # Download progress based on status
        if status == "ready":
            progress = 100.0
        elif status == "downloading":
            progress = 50.0
        elif status == "starting":
            progress = 75.0  # Model cached, loading into GPU
        else:
            progress = 0.0

        model_files.append(
            ModelFileView(
                model_id=mid,
                worker_id=wid,
                model_name=first_dep.model.name if first_dep.model else "Unknown",
                model_source=first_dep.model.model_id if first_dep.model else "",
                worker_name=first_dep.worker.name if first_dep.worker else "Unknown",
                worker_address=first_dep.worker.address if first_dep.worker else "",
                status=status,
                download_progress=progress,
                deployment_count=len(deps),
                running_count=running_count,
                deployments=[
                    ModelFileDeployment(
                        id=d.id,
                        name=d.name,
                        status=d.status,
                        port=d.port,
                    )
                    for d in sorted(deps, key=lambda x: x.created_at, reverse=True)
                ],
            )
        )

    # Sort by model name, then worker name for consistent ordering
    model_files.sort(key=lambda x: (x.model_name.lower(), x.worker_name.lower()))

    # Apply pagination
    total = len(model_files)
    model_files = model_files[skip : skip + limit]

    return ModelFileListResponse(items=model_files, total=total)


@router.delete("/{model_id}/workers/{worker_id}")
async def delete_model_file(
    model_id: int,
    worker_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete model files from a specific worker.

    This operation removes all stopped deployments of the specified model
    on the specified worker. Active (running/starting/downloading) deployments
    must be stopped first before deletion.

    Returns:
    - 200: Successfully deleted deployments
    - 400: Cannot delete - active deployments exist
    - 404: No deployments found for this model/worker combination
    """
    # Find all deployments for this model on this worker
    result = await db.execute(
        select(Deployment)
        .where(Deployment.model_id == model_id)
        .where(Deployment.worker_id == worker_id)
    )
    deployments = result.scalars().all()

    if not deployments:
        raise HTTPException(
            status_code=404, detail="No model files found for this model on this worker"
        )

    # Check for active deployments
    active_statuses = [
        DeploymentStatus.RUNNING.value,
        DeploymentStatus.STARTING.value,
        DeploymentStatus.DOWNLOADING.value,
        DeploymentStatus.PENDING.value,
    ]
    active = [d for d in deployments if d.status in active_statuses]

    if active:
        active_names = ", ".join(d.name for d in active[:3])
        if len(active) > 3:
            active_names += f" and {len(active) - 3} more"
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete: {len(active)} deployment(s) are still active ({active_names}). Stop them first.",
        )

    # Delete all stopped deployments
    deleted_count = 0
    for d in deployments:
        await db.delete(d)
        deleted_count += 1

    await db.commit()

    return {
        "message": f"Successfully deleted {deleted_count} deployment(s)",
        "deleted_count": deleted_count,
    }

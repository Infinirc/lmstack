"""Worker API routes"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.deployment import Deployment
from app.models.worker import Worker, WorkerStatus
from app.schemas.worker import (
    WorkerCreate,
    WorkerHeartbeat,
    WorkerListResponse,
    WorkerResponse,
    WorkerUpdate,
)
from app.services.local_worker import get_local_worker_info

router = APIRouter()


@router.get("", response_model=WorkerListResponse)
async def list_workers(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List all workers"""
    query = select(Worker)

    if status:
        query = query.where(Worker.status == status)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Get paginated results
    query = query.offset(skip).limit(limit).order_by(Worker.created_at.desc())
    result = await db.execute(query)
    workers = result.scalars().all()

    # Add deployment count
    items = []
    for worker in workers:
        deployment_count_query = select(func.count()).where(Deployment.worker_id == worker.id)
        deployment_count = await db.scalar(deployment_count_query) or 0

        worker_dict = {
            "id": worker.id,
            "name": worker.name,
            "address": worker.address,
            "description": worker.description,
            "labels": worker.labels,
            "status": worker.status,
            "gpu_info": worker.gpu_info,
            "system_info": worker.system_info,
            "created_at": worker.created_at,
            "updated_at": worker.updated_at,
            "last_heartbeat": worker.last_heartbeat,
            "deployment_count": deployment_count,
        }
        items.append(WorkerResponse(**worker_dict))

    return WorkerListResponse(items=items, total=total or 0)


@router.post("", response_model=WorkerResponse, status_code=201)
async def create_worker(
    worker_in: WorkerCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new worker"""
    # Check if worker with same name exists
    existing = await db.execute(select(Worker).where(Worker.name == worker_in.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Worker with this name already exists")

    worker = Worker(
        name=worker_in.name,
        address=worker_in.address,
        description=worker_in.description,
        labels=worker_in.labels,
        gpu_info=([gpu.model_dump() for gpu in worker_in.gpu_info] if worker_in.gpu_info else None),
        system_info=(worker_in.system_info.model_dump() if worker_in.system_info else None),
        status=WorkerStatus.ONLINE.value,
        last_heartbeat=datetime.now(UTC),
    )

    db.add(worker)
    await db.commit()
    await db.refresh(worker)

    return WorkerResponse(
        id=worker.id,
        name=worker.name,
        address=worker.address,
        description=worker.description,
        labels=worker.labels,
        status=worker.status,
        gpu_info=worker.gpu_info,
        system_info=worker.system_info,
        created_at=worker.created_at,
        updated_at=worker.updated_at,
        last_heartbeat=worker.last_heartbeat,
        deployment_count=0,
    )


@router.get("/{worker_id}", response_model=WorkerResponse)
async def get_worker(
    worker_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a worker by ID"""
    result = await db.execute(select(Worker).where(Worker.id == worker_id))
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    deployment_count_query = select(func.count()).where(Deployment.worker_id == worker.id)
    deployment_count = await db.scalar(deployment_count_query) or 0

    return WorkerResponse(
        id=worker.id,
        name=worker.name,
        address=worker.address,
        description=worker.description,
        labels=worker.labels,
        status=worker.status,
        gpu_info=worker.gpu_info,
        system_info=worker.system_info,
        created_at=worker.created_at,
        updated_at=worker.updated_at,
        last_heartbeat=worker.last_heartbeat,
        deployment_count=deployment_count,
    )


@router.patch("/{worker_id}", response_model=WorkerResponse)
async def update_worker(
    worker_id: int,
    worker_in: WorkerUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a worker"""
    result = await db.execute(select(Worker).where(Worker.id == worker_id))
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    update_data = worker_in.model_dump(exclude_unset=True)

    if "gpu_info" in update_data and update_data["gpu_info"]:
        update_data["gpu_info"] = [gpu.model_dump() for gpu in worker_in.gpu_info]

    if "system_info" in update_data and update_data["system_info"]:
        update_data["system_info"] = worker_in.system_info.model_dump()

    if "status" in update_data and update_data["status"]:
        update_data["status"] = update_data["status"].value

    for field, value in update_data.items():
        setattr(worker, field, value)

    await db.commit()
    await db.refresh(worker)

    deployment_count_query = select(func.count()).where(Deployment.worker_id == worker.id)
    deployment_count = await db.scalar(deployment_count_query) or 0

    return WorkerResponse(
        id=worker.id,
        name=worker.name,
        address=worker.address,
        description=worker.description,
        labels=worker.labels,
        status=worker.status,
        gpu_info=worker.gpu_info,
        system_info=worker.system_info,
        created_at=worker.created_at,
        updated_at=worker.updated_at,
        last_heartbeat=worker.last_heartbeat,
        deployment_count=deployment_count,
    )


@router.delete("/{worker_id}", status_code=204)
async def delete_worker(
    worker_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a worker"""
    result = await db.execute(select(Worker).where(Worker.id == worker_id))
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Check if worker has active deployments
    deployment_count = await db.scalar(
        select(func.count()).where(Deployment.worker_id == worker_id)
    )
    if deployment_count and deployment_count > 0:
        raise HTTPException(status_code=400, detail="Cannot delete worker with active deployments")

    await db.delete(worker)
    await db.commit()


@router.post("/heartbeat")
async def worker_heartbeat(
    heartbeat: WorkerHeartbeat,
    db: AsyncSession = Depends(get_db),
):
    """Receive heartbeat from worker"""
    result = await db.execute(select(Worker).where(Worker.id == heartbeat.worker_id))
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    worker.last_heartbeat = datetime.now(UTC)
    worker.status = heartbeat.status.value

    if heartbeat.gpu_info:
        worker.gpu_info = [gpu.model_dump() for gpu in heartbeat.gpu_info]

    if heartbeat.system_info:
        worker.system_info = heartbeat.system_info.model_dump()

    await db.commit()

    return {"status": "ok"}


@router.post("/local", response_model=WorkerResponse, status_code=201)
async def register_local_worker(
    db: AsyncSession = Depends(get_db),
):
    """Register the local machine as a worker"""
    # Get local machine info
    info = get_local_worker_info()

    worker_name = f"local-{info['hostname']}"

    # Check if local worker already exists
    existing = await db.execute(select(Worker).where(Worker.name == worker_name))
    existing_worker = existing.scalar_one_or_none()

    if existing_worker:
        # Update existing worker info
        existing_worker.gpu_info = info["gpu_info"]
        existing_worker.system_info = info["system_info"]
        existing_worker.status = WorkerStatus.ONLINE.value
        existing_worker.last_heartbeat = datetime.now(UTC)

        await db.commit()
        await db.refresh(existing_worker)

        deployment_count_query = select(func.count()).where(
            Deployment.worker_id == existing_worker.id
        )
        deployment_count = await db.scalar(deployment_count_query) or 0

        return WorkerResponse(
            id=existing_worker.id,
            name=existing_worker.name,
            address=existing_worker.address,
            description=existing_worker.description,
            labels=existing_worker.labels,
            status=existing_worker.status,
            gpu_info=existing_worker.gpu_info,
            system_info=existing_worker.system_info,
            created_at=existing_worker.created_at,
            updated_at=existing_worker.updated_at,
            last_heartbeat=existing_worker.last_heartbeat,
            deployment_count=deployment_count,
        )

    # Create new local worker
    worker = Worker(
        name=worker_name,
        address="localhost",
        description=f"Local worker on {info['hostname']} ({info['platform']} {info['platform_release']})",
        labels={"type": "local", "hostname": info["hostname"]},
        gpu_info=info["gpu_info"],
        system_info=info["system_info"],
        status=WorkerStatus.ONLINE.value,
        last_heartbeat=datetime.now(UTC),
    )

    db.add(worker)
    await db.commit()
    await db.refresh(worker)

    return WorkerResponse(
        id=worker.id,
        name=worker.name,
        address=worker.address,
        description=worker.description,
        labels=worker.labels,
        status=worker.status,
        gpu_info=worker.gpu_info,
        system_info=worker.system_info,
        created_at=worker.created_at,
        updated_at=worker.updated_at,
        last_heartbeat=worker.last_heartbeat,
        deployment_count=0,
    )

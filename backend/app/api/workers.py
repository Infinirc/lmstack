"""Worker API routes"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.deps import require_operator, require_viewer
from app.database import get_db
from app.models.deployment import Deployment
from app.models.registration_token import RegistrationToken
from app.models.user import User
from app.models.worker import Worker, WorkerStatus
from app.schemas.worker import (
    RegistrationTokenCreate,
    RegistrationTokenListResponse,
    RegistrationTokenResponse,
    WorkerCreate,
    WorkerHeartbeat,
    WorkerListResponse,
    WorkerRegisterWithToken,
    WorkerResponse,
    WorkerUpdate,
)
from app.services.local_worker import (
    get_local_hostname,
    get_local_ip,
    spawn_docker_worker,
    stop_docker_worker,
)

router = APIRouter()


@router.get("", response_model=WorkerListResponse)
async def list_workers(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """List all workers (requires viewer+)"""
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


def _get_client_ip(request: Request) -> str:
    """Get real client IP from request headers or connection."""
    # Check X-Forwarded-For first (from proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    # Check X-Real-IP (from nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    # Fall back to direct connection IP
    if request.client:
        return request.client.host
    return "127.0.0.1"


@router.post("", response_model=WorkerResponse, status_code=201)
async def create_worker(
    worker_in: WorkerCreate | WorkerRegisterWithToken,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Register a new worker (requires registration token)"""
    # Check for registration token
    registration_token = getattr(worker_in, "registration_token", None)
    if not registration_token:
        raise HTTPException(
            status_code=401,
            detail="Registration token required. Generate a token from the web UI.",
        )

    # Verify token
    result = await db.execute(
        select(RegistrationToken).where(RegistrationToken.token == registration_token)
    )
    token = result.scalar_one_or_none()

    if not token:
        raise HTTPException(status_code=401, detail="Invalid registration token")

    # Check if worker with same name exists
    existing_result = await db.execute(select(Worker).where(Worker.name == worker_in.name))
    existing_worker = existing_result.scalar_one_or_none()

    # If token is used, check if it's for reconnecting the same worker or if worker was deleted
    if token.is_used:
        # Check if the original worker still exists
        original_worker_result = await db.execute(
            select(Worker).where(Worker.id == token.used_by_worker_id)
        )
        original_worker = original_worker_result.scalar_one_or_none()

        if existing_worker and token.used_by_worker_id == existing_worker.id:
            # Allow reconnection - update existing worker with real IP
            client_ip = _get_client_ip(request)
            reported_port = "52001"
            if ":" in worker_in.address:
                reported_port = worker_in.address.split(":")[-1]
            existing_worker.address = f"{client_ip}:{reported_port}"
            existing_worker.gpu_info = (
                [gpu.model_dump() for gpu in worker_in.gpu_info] if worker_in.gpu_info else None
            )
            existing_worker.system_info = (
                worker_in.system_info.model_dump() if worker_in.system_info else None
            )
            existing_worker.status = WorkerStatus.ONLINE.value
            existing_worker.last_heartbeat = datetime.now(UTC)

            await db.commit()
            await db.refresh(existing_worker)

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
                deployment_count=0,
            )
        elif original_worker is None:
            # Original worker was deleted, allow re-registration with new worker
            # Reset token for reuse
            token.is_used = False
            token.used_at = None
            token.used_by_worker_id = None
            await db.commit()
            # Continue to create new worker below
        else:
            raise HTTPException(status_code=401, detail="Registration token has already been used")

    if not token.is_valid:
        raise HTTPException(status_code=401, detail="Registration token has expired")

    if existing_worker:
        raise HTTPException(status_code=400, detail="Worker with this name already exists")

    # Use real client IP instead of reported address (which might be Docker internal IP)
    client_ip = _get_client_ip(request)
    # Extract port from reported address
    reported_port = "52001"
    if ":" in worker_in.address:
        reported_port = worker_in.address.split(":")[-1]
    real_address = f"{client_ip}:{reported_port}"

    worker = Worker(
        name=worker_in.name,
        address=real_address,
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

    # Mark token as used
    token.is_used = True
    token.used_at = datetime.now(UTC)
    token.used_by_worker_id = worker.id
    await db.commit()

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
    current_user: User = Depends(require_viewer),
):
    """Get a worker by ID (requires viewer+)"""
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
    current_user: User = Depends(require_operator),
):
    """Update a worker (requires operator+)"""
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
    current_user: User = Depends(require_operator),
):
    """Delete a worker (requires operator+)"""
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

    # Try to stop and remove Docker container if it's a local worker
    # Container name is "lmstack-worker" by default
    stop_docker_worker("lmstack-worker")

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


@router.post("/local", status_code=201)
async def register_local_worker(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Spawn a Docker worker on the local machine (requires operator+).

    This creates a registration token, then runs docker to start a worker container.
    The worker will register itself using the token.
    """
    # Get local hostname for worker name
    hostname = get_local_hostname()
    worker_name = hostname

    # Get backend URL - use local IP so the Docker container can reach it
    local_ip = get_local_ip()
    settings = get_settings()
    backend_url = f"http://{local_ip}:{settings.port}"

    # Create a registration token for this worker
    token = RegistrationToken.create(
        name=worker_name,
        expires_in_hours=24,  # Token valid for 24 hours
    )

    db.add(token)
    await db.commit()
    await db.refresh(token)

    # Spawn the Docker worker
    result = spawn_docker_worker(
        worker_name=worker_name,
        backend_url=backend_url,
        registration_token=token.token,
        container_name="lmstack-worker",
    )

    if not result["success"]:
        # Clean up the token if spawn failed
        await db.delete(token)
        await db.commit()
        raise HTTPException(status_code=500, detail=result["message"])

    return {
        "success": True,
        "message": result["message"],
        "worker_name": worker_name,
        "container_id": result.get("container_id"),
        "backend_url": backend_url,
    }


def _generate_docker_command(token: str, name: str, backend_url: str) -> str:
    """Generate docker run command for worker registration."""
    return f"""docker run -d \\
  --name lmstack-worker \\
  --gpus all \\
  --privileged \\
  -p 52001:52001 \\
  -v /var/run/docker.sock:/var/run/docker.sock \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  -v /:/host:ro \\
  -e BACKEND_URL={backend_url} \\
  -e WORKER_NAME={name} \\
  -e REGISTRATION_TOKEN={token} \\
  infinirc/lmstack-worker:latest"""


def _get_backend_url(request: Request) -> str:
    """Get backend URL for worker registration."""
    settings = get_settings()
    if settings.external_url:
        return settings.external_url.rstrip("/")
    # Check X-Forwarded headers (from nginx/vite proxy)
    forwarded_host = request.headers.get("X-Forwarded-Host")
    if forwarded_host:
        proto = request.headers.get("X-Forwarded-Proto", "http")
        # Replace port with backend's actual port (for dev mode where frontend is on different port)
        host_parts = forwarded_host.split(":")
        hostname = host_parts[0]
        return f"{proto}://{hostname}:{settings.port}"
    # Fallback to request URL
    return str(request.base_url).rstrip("/")


@router.post("/tokens", response_model=RegistrationTokenResponse, status_code=201)
async def create_registration_token(
    token_in: RegistrationTokenCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Generate a new registration token for worker registration (requires operator+)."""
    token = RegistrationToken.create(
        name=token_in.name,
        expires_in_hours=token_in.expires_in_hours,
    )

    db.add(token)
    await db.commit()
    await db.refresh(token)

    backend_url = _get_backend_url(request)

    return RegistrationTokenResponse(
        id=token.id,
        token=token.token,
        name=token.name,
        is_used=token.is_used,
        used_by_worker_id=token.used_by_worker_id,
        created_at=token.created_at,
        expires_at=token.expires_at,
        used_at=token.used_at,
        is_valid=token.is_valid,
        docker_command=_generate_docker_command(token.token, token.name, backend_url),
    )


@router.get("/tokens", response_model=RegistrationTokenListResponse)
async def list_registration_tokens(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    include_used: bool = Query(False, description="Include used tokens"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """List registration tokens (requires viewer+)."""
    query = select(RegistrationToken)

    if not include_used:
        query = query.where(RegistrationToken.is_used == False)  # noqa: E712

    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    query = query.offset(skip).limit(limit).order_by(RegistrationToken.created_at.desc())
    result = await db.execute(query)
    tokens = result.scalars().all()

    items = [
        RegistrationTokenResponse(
            id=t.id,
            token=t.token,
            name=t.name,
            is_used=t.is_used,
            used_by_worker_id=t.used_by_worker_id,
            created_at=t.created_at,
            expires_at=t.expires_at,
            used_at=t.used_at,
            is_valid=t.is_valid,
            docker_command=None,
        )
        for t in tokens
    ]

    return RegistrationTokenListResponse(items=items, total=total or 0)


@router.get("/tokens/{token_id}", response_model=RegistrationTokenResponse)
async def get_registration_token(
    token_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Get a registration token by ID (requires viewer+)."""
    result = await db.execute(select(RegistrationToken).where(RegistrationToken.id == token_id))
    token = result.scalar_one_or_none()

    if not token:
        raise HTTPException(status_code=404, detail="Registration token not found")

    backend_url = _get_backend_url(request)

    return RegistrationTokenResponse(
        id=token.id,
        token=token.token,
        name=token.name,
        is_used=token.is_used,
        used_by_worker_id=token.used_by_worker_id,
        created_at=token.created_at,
        expires_at=token.expires_at,
        used_at=token.used_at,
        is_valid=token.is_valid,
        docker_command=(
            _generate_docker_command(token.token, token.name, backend_url)
            if token.is_valid
            else None
        ),
    )


@router.delete("/tokens/{token_id}", status_code=204)
async def delete_registration_token(
    token_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Delete a registration token (requires operator+)."""
    result = await db.execute(select(RegistrationToken).where(RegistrationToken.id == token_id))
    token = result.scalar_one_or_none()

    if not token:
        raise HTTPException(status_code=404, detail="Registration token not found")

    await db.delete(token)
    await db.commit()

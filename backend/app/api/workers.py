"""Worker API routes"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.deps import require_operator, require_viewer
from app.database import async_session_maker, get_db
from app.models.app import App, AppStatus
from app.models.deployment import Deployment, DeploymentStatus
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
from app.services.local_worker import get_local_hostname, spawn_docker_worker, stop_docker_worker

logger = logging.getLogger(__name__)

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
            "os_type": worker.os_type,
            "gpu_type": worker.gpu_type,
            "capabilities": worker.capabilities,
            "available_backends": worker.available_backends,
            "connection_type": worker.connection_type,
            "tailscale_ip": worker.tailscale_ip,
            "headscale_node_id": worker.headscale_node_id,
            "effective_address": worker.effective_address,
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
    background_tasks: BackgroundTasks,
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

        if original_worker is not None:
            # Allow reconnection - update existing worker with new info
            # Worker name may have changed (e.g., new container ID), update it
            client_ip = _get_client_ip(request)
            reported_port = "52001"
            if ":" in worker_in.address:
                reported_port = worker_in.address.split(":")[-1]
            original_worker.name = worker_in.name  # Update name in case it changed
            original_worker.address = f"{client_ip}:{reported_port}"
            original_worker.gpu_info = (
                [gpu.model_dump() for gpu in worker_in.gpu_info] if worker_in.gpu_info else None
            )
            original_worker.system_info = (
                worker_in.system_info.model_dump() if worker_in.system_info else None
            )
            # Update os_type, gpu_type, capabilities from system_info
            if worker_in.system_info:
                if worker_in.system_info.os_type:
                    original_worker.os_type = worker_in.system_info.os_type
                if worker_in.system_info.gpu_type:
                    original_worker.gpu_type = worker_in.system_info.gpu_type
                if worker_in.system_info.capabilities:
                    original_worker.capabilities = worker_in.system_info.capabilities.model_dump()
            original_worker.status = WorkerStatus.ONLINE.value
            original_worker.last_heartbeat = datetime.now(UTC)

            # Update labels to mark as local if token is for local worker
            if token.is_local:
                worker_labels = dict(original_worker.labels) if original_worker.labels else {}
                worker_labels["type"] = "local"
                original_worker.labels = worker_labels

            await db.commit()
            await db.refresh(original_worker)

            # Refresh deployments and apps status on this worker
            background_tasks.add_task(_refresh_worker_resources, original_worker.id)

            return WorkerResponse(
                id=original_worker.id,
                name=original_worker.name,
                address=original_worker.address,
                description=original_worker.description,
                labels=original_worker.labels,
                status=original_worker.status,
                gpu_info=original_worker.gpu_info,
                system_info=original_worker.system_info,
                os_type=original_worker.os_type,
                gpu_type=original_worker.gpu_type,
                capabilities=original_worker.capabilities,
                available_backends=original_worker.available_backends,
                connection_type=original_worker.connection_type,
                tailscale_ip=original_worker.tailscale_ip,
                headscale_node_id=original_worker.headscale_node_id,
                effective_address=original_worker.effective_address,
                created_at=original_worker.created_at,
                updated_at=original_worker.updated_at,
                last_heartbeat=original_worker.last_heartbeat,
                deployment_count=0,
            )
        else:
            # Original worker was deleted, allow re-registration with new worker
            # Reset token for reuse
            token.is_used = False
            token.used_at = None
            token.used_by_worker_id = None
            await db.commit()
            # Continue to create new worker below

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

    # Set labels for local workers (created via /local endpoint)
    worker_labels = dict(worker_in.labels) if worker_in.labels else {}
    if token.is_local:
        worker_labels["type"] = "local"

    # Extract os_type, gpu_type, capabilities from system_info
    os_type = "linux"
    gpu_type = "nvidia"
    capabilities = None
    if worker_in.system_info:
        if worker_in.system_info.os_type:
            os_type = worker_in.system_info.os_type
        if worker_in.system_info.gpu_type:
            gpu_type = worker_in.system_info.gpu_type
        if worker_in.system_info.capabilities:
            capabilities = worker_in.system_info.capabilities.model_dump()

    worker = Worker(
        name=worker_in.name,
        address=real_address,
        description=worker_in.description,
        labels=worker_labels if worker_labels else None,
        gpu_info=([gpu.model_dump() for gpu in worker_in.gpu_info] if worker_in.gpu_info else None),
        system_info=(worker_in.system_info.model_dump() if worker_in.system_info else None),
        os_type=os_type,
        gpu_type=gpu_type,
        capabilities=capabilities,
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
        os_type=worker.os_type,
        gpu_type=worker.gpu_type,
        capabilities=worker.capabilities,
        available_backends=worker.available_backends,
        connection_type=worker.connection_type,
        tailscale_ip=worker.tailscale_ip,
        headscale_node_id=worker.headscale_node_id,
        effective_address=worker.effective_address,
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
        os_type=worker.os_type,
        gpu_type=worker.gpu_type,
        capabilities=worker.capabilities,
        available_backends=worker.available_backends,
        connection_type=worker.connection_type,
        tailscale_ip=worker.tailscale_ip,
        headscale_node_id=worker.headscale_node_id,
        effective_address=worker.effective_address,
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
        os_type=worker.os_type,
        gpu_type=worker.gpu_type,
        capabilities=worker.capabilities,
        available_backends=worker.available_backends,
        connection_type=worker.connection_type,
        tailscale_ip=worker.tailscale_ip,
        headscale_node_id=worker.headscale_node_id,
        effective_address=worker.effective_address,
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
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Receive heartbeat from worker"""
    result = await db.execute(select(Worker).where(Worker.id == heartbeat.worker_id))
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Check if worker is coming back online
    was_offline = worker.status == WorkerStatus.OFFLINE.value
    is_now_online = heartbeat.status == WorkerStatus.ONLINE

    worker.last_heartbeat = datetime.now(UTC)
    worker.status = heartbeat.status.value

    if heartbeat.gpu_info:
        worker.gpu_info = [gpu.model_dump() for gpu in heartbeat.gpu_info]

    if heartbeat.system_info:
        system_data = heartbeat.system_info.model_dump()
        worker.system_info = system_data
        # Extract os_type, gpu_type, capabilities from system_info
        if heartbeat.system_info.os_type:
            worker.os_type = heartbeat.system_info.os_type
        if heartbeat.system_info.gpu_type:
            worker.gpu_type = heartbeat.system_info.gpu_type
        if heartbeat.system_info.capabilities:
            worker.capabilities = heartbeat.system_info.capabilities.model_dump()

    # Check if worker is going offline
    is_going_offline = heartbeat.status == WorkerStatus.OFFLINE

    # If worker is going offline, immediately update all deployments and apps
    if is_going_offline:
        await _mark_worker_resources_offline(db, worker.id, worker.name)

    await db.commit()

    # If worker came back online, refresh deployments and apps status
    if was_offline and is_now_online:
        background_tasks.add_task(_refresh_worker_resources, worker.id)

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

    # Use localhost since local worker uses --network host mode
    settings = get_settings()
    backend_url = f"http://localhost:{settings.port}"

    # Create a registration token for this worker (marked as local)
    token = RegistrationToken.create(
        name=worker_name,
        expires_in_hours=24,  # Token valid for 24 hours
        is_local=True,  # Mark as local worker
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
    """Generate docker run command for worker registration.

    Uses --network host mode so that:
    1. Worker registers with localhost/host IP (not Docker internal IP)
    2. Apps deployed by Worker are accessible via host network
    3. Works seamlessly on both regular machines and WSL

    Uses --restart unless-stopped so worker auto-starts after system reboot.

    Command is single-line for cross-platform compatibility (Linux/Mac/Windows).
    """
    return (
        f"docker run -d --name lmstack-worker --restart unless-stopped "
        f"--network host --gpus all --privileged "
        f"-v /var/run/docker.sock:/var/run/docker.sock "
        f"-v ~/.cache/huggingface:/root/.cache/huggingface "
        f"-v /:/host:ro "
        f"-e BACKEND_URL={backend_url} "
        f"-e WORKER_NAME={name} "
        f"-e REGISTRATION_TOKEN={token} "
        f"infinirc/lmstack-worker:latest"
    )


def _get_wsl_windows_ip() -> str | None:
    """Get Windows host IP from WSL."""
    try:
        # Check if we're in WSL
        with open("/proc/version") as f:
            if "microsoft" not in f.read().lower():
                return None
        # Read Windows host IP from resolv.conf (nameserver is Windows host)
        with open("/etc/resolv.conf") as f:
            for line in f:
                if line.startswith("nameserver"):
                    ip = line.split()[1].strip()
                    # Filter out localhost entries
                    if not ip.startswith("127."):
                        return ip
    except Exception:
        pass
    return None


def _get_backend_url(request: Request) -> str:
    """Get backend URL for worker registration."""
    settings = get_settings()
    if settings.external_url:
        return settings.external_url.rstrip("/")

    # For WSL, try to get the Windows host IP for external access
    windows_ip = _get_wsl_windows_ip()
    if windows_ip:
        return f"http://{windows_ip}:{settings.port}"

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


# =============================================================================
# Helper Functions
# =============================================================================


async def _mark_worker_resources_offline(db: AsyncSession, worker_id: int, worker_name: str):
    """Mark all deployments and apps on an offline worker as unavailable.

    This is called synchronously when a worker sends an offline heartbeat.
    """
    # Update deployments on this worker
    dep_result = await db.execute(
        select(Deployment).where(
            Deployment.worker_id == worker_id,
            Deployment.status.in_(
                [
                    DeploymentStatus.RUNNING.value,
                    DeploymentStatus.STARTING.value,
                ]
            ),
        )
    )
    deployments = dep_result.scalars().all()

    for deployment in deployments:
        deployment.status = DeploymentStatus.ERROR.value
        deployment.status_message = f"Worker {worker_name} is offline"

    # Update apps on this worker
    app_result = await db.execute(
        select(App).where(
            App.worker_id == worker_id,
            App.status.in_(
                [
                    AppStatus.RUNNING.value,
                    AppStatus.STARTING.value,
                    AppStatus.PULLING.value,
                ]
            ),
        )
    )
    apps = app_result.scalars().all()

    for app in apps:
        app.status = AppStatus.ERROR.value
        app.status_message = f"Worker {worker_name} is offline"

    if deployments or apps:
        logger.info(
            f"Marked {len(deployments)} deployments and {len(apps)} apps as offline "
            f"for worker {worker_name}"
        )


async def _refresh_worker_resources(worker_id: int):
    """Refresh status of deployments and apps on a worker that just came online.

    This is called as a background task when a worker's heartbeat indicates
    it has come back online after being offline.
    """
    import httpx

    logger.info(f"Refreshing resources for worker {worker_id} after coming online")

    async with async_session_maker() as db:
        # Get the worker
        result = await db.execute(select(Worker).where(Worker.id == worker_id))
        worker = result.scalar_one_or_none()
        if not worker:
            return

        # Refresh deployments on this worker
        dep_result = await db.execute(
            select(Deployment).where(
                Deployment.worker_id == worker_id,
                Deployment.status.in_(
                    [
                        DeploymentStatus.ERROR.value,
                        DeploymentStatus.STARTING.value,
                        DeploymentStatus.RUNNING.value,
                    ]
                ),
            )
        )
        deployments = dep_result.scalars().all()

        for deployment in deployments:
            if not deployment.container_id:
                continue
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Check if this is a native deployment
                    if deployment.container_id.startswith("native-"):
                        # For native deployments, check via /native/processes
                        response = await client.get(f"http://{worker.address}/native/processes")
                        if response.status_code == 200:
                            processes = response.json().get("processes", [])
                            found = False
                            for p in processes:
                                if p.get("process_id") == deployment.container_id:
                                    found = True
                                    if p.get("running"):
                                        deployment.status = DeploymentStatus.RUNNING.value
                                        deployment.status_message = "Model ready"
                                    else:
                                        deployment.status = DeploymentStatus.STOPPED.value
                                        deployment.status_message = "Process stopped"
                                    break
                            if not found:
                                # Process not in manager, but might still be running via Ollama
                                # Don't mark as error, just skip
                                pass
                    else:
                        # Docker container check
                        response = await client.get(
                            f"http://{worker.address}/containers/{deployment.container_id}"
                        )
                        if response.status_code == 200:
                            container_info = response.json()
                            state = container_info.get("state", "").lower()
                            if state == "running":
                                deployment.status = DeploymentStatus.RUNNING.value
                                deployment.status_message = "Model ready"
                            elif state in ("exited", "dead"):
                                deployment.status = DeploymentStatus.STOPPED.value
                                deployment.status_message = f"Container {state}"
                        elif response.status_code == 404:
                            deployment.status = DeploymentStatus.ERROR.value
                            deployment.status_message = "Container not found"
            except Exception as e:
                logger.warning(f"Failed to check deployment {deployment.id}: {e}")

        # Refresh apps on this worker
        app_result = await db.execute(
            select(App).where(
                App.worker_id == worker_id,
                App.status.in_(
                    [
                        AppStatus.ERROR.value,
                        AppStatus.STARTING.value,
                        AppStatus.RUNNING.value,
                    ]
                ),
            )
        )
        apps = app_result.scalars().all()

        for app in apps:
            if not app.container_id:
                continue
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"http://{worker.address}/containers/{app.container_id}"
                    )
                    if response.status_code == 200:
                        container_info = response.json()
                        state = container_info.get("state", "").lower()
                        if state == "running":
                            app.status = AppStatus.RUNNING.value
                            app.status_message = None
                        elif state in ("exited", "dead"):
                            app.status = AppStatus.STOPPED.value
                            app.status_message = f"Container {state}"
                    elif response.status_code == 404:
                        app.status = AppStatus.ERROR.value
                        app.status_message = "Container not found"
            except Exception as e:
                logger.warning(f"Failed to check app {app.id}: {e}")

        await db.commit()
        logger.info(
            f"Refreshed {len(deployments)} deployments and {len(apps)} apps for worker {worker_id}"
        )

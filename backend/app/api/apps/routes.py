"""API routes for Deploy Apps.

Contains the main FastAPI endpoint handlers for app deployment.
Lifecycle endpoints (start/stop/delete/logs) are in lifecycle.py.
"""

import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, BackgroundTasks
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.app import App, AppType, AppStatus, APP_DEFINITIONS
from app.models.worker import Worker
from app.models.api_key import ApiKey
from app.models.deployment import Deployment
from app.models.llm_model import LLMModel
from app.schemas.app import (
    AppDefinition,
    AppDeploy,
    AppResponse,
    AppListResponse,
    AvailableAppsResponse,
)
from app.api.apps.utils import (
    API_KEY_PREFIX,
    generate_access_key,
    generate_secret_key,
    hash_secret,
    get_worker_or_404,
    app_to_response,
    get_host_ip,
)
from app.api.apps.deployment import (
    deploy_app_background,
    get_deployment_progress,
    set_deployment_progress,
)
from app.api.apps.lifecycle import router as lifecycle_router

logger = logging.getLogger(__name__)
router = APIRouter()

# Include lifecycle routes (start/stop/delete/logs)
router.include_router(lifecycle_router)


# =============================================================================
# List & Discovery Endpoints
# =============================================================================


@router.get("/available", response_model=AvailableAppsResponse)
async def list_available_apps():
    """List all available apps that can be deployed."""
    items = []
    for app_type, definition in APP_DEFINITIONS.items():
        items.append(
            AppDefinition(
                type=app_type.value,
                name=definition["name"],
                description=definition["description"],
                image=definition["image"],
            )
        )
    return AvailableAppsResponse(items=items)


@router.get("", response_model=AppListResponse)
async def list_apps(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List all deployed apps."""
    # Count total
    total = await db.scalar(select(func.count()).select_from(App))

    # Get paginated results with worker relationship
    result = await db.execute(select(App).offset(skip).limit(limit).order_by(App.created_at.desc()))
    apps = result.scalars().all()

    # Load worker relationships
    for app in apps:
        await db.refresh(app, ["worker"])

    return AppListResponse(
        items=[app_to_response(app, request) for app in apps],
        total=total or 0,
    )


@router.get("/{app_id}/progress")
async def get_app_deploy_progress(app_id: int):
    """Get deployment progress for an app."""
    return get_deployment_progress(app_id)


@router.get("/{app_id}", response_model=AppResponse)
async def get_app(
    app_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get app details."""
    result = await db.execute(select(App).where(App.id == app_id))
    app = result.scalar_one_or_none()

    if not app:
        raise HTTPException(status_code=404, detail="App not found")

    await db.refresh(app, ["worker"])
    return app_to_response(app, request)


# =============================================================================
# Deploy Endpoint
# =============================================================================


@router.post("", response_model=AppResponse, status_code=201)
async def deploy_app(
    deploy_request: AppDeploy,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Deploy a new app.

    This will:
    1. Create an API key for the app to access LMStack
    2. Start a background task to pull the image and start the container
    3. Return immediately with the app in "pending" status
    """
    # Validate app type
    try:
        app_type = AppType(deploy_request.app_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid app type: {deploy_request.app_type}")

    app_def = APP_DEFINITIONS[app_type]

    # Get worker
    worker = await get_worker_or_404(deploy_request.worker_id, db)

    # Check if app of this type already exists on this worker
    existing = await db.execute(
        select(App).where(
            App.app_type == app_type.value,
            App.worker_id == worker.id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=400, detail=f"{app_def['name']} is already deployed on worker {worker.name}"
        )

    # Generate app name
    app_name = deploy_request.name or f"{app_def['name']}"

    # Create API key for apps that need it
    api_key, full_key = await _create_api_key_if_needed(db, app_name, app_def)

    # Create proxy path and find available port
    proxy_path = f"/apps/{app_type.value}"
    port = await _find_available_port(db, worker.id)

    # Create app record
    app = App(
        app_type=app_type.value,
        name=app_name,
        worker_id=worker.id,
        api_key_id=api_key.id if api_key else None,
        status=AppStatus.PENDING.value,
        proxy_path=proxy_path,
        port=port,
        use_proxy=deploy_request.use_proxy,
    )
    db.add(app)
    await db.commit()
    await db.refresh(app, ["worker"])

    # Build environment variables
    env_vars = await _build_env_vars(
        request=request,
        worker=worker,
        app_def=app_def,
        full_key=full_key,
        port=port,
        db=db,
    )

    # Initialize progress
    set_deployment_progress(app.id, "pending", 0, "Deployment queued...")

    # Extract lmstack_port for background task
    lmstack_host = request.headers.get("host", "localhost:8000")
    lmstack_port = lmstack_host.split(":")[-1] if ":" in lmstack_host else "8000"

    # Start background deployment
    background_tasks.add_task(
        deploy_app_background,
        app_id=app.id,
        app_type=app_type,
        worker_address=worker.address,
        worker_host=worker.address.split(":")[0],
        env_vars=env_vars,
        port=port,
        app_def=app_def,
        lmstack_port=lmstack_port,
        use_proxy=deploy_request.use_proxy,
    )

    return app_to_response(app, request)


# =============================================================================
# Helper Functions
# =============================================================================


async def _create_api_key_if_needed(
    db: AsyncSession,
    app_name: str,
    app_def: dict,
) -> tuple[Optional[ApiKey], str]:
    """Create API key for apps that need it.

    Returns:
        Tuple of (ApiKey or None, full_key string)
    """
    needs_api_key = any(v == "{api_key}" for v in app_def.get("env_template", {}).values())

    if not needs_api_key:
        return None, ""

    access_key = generate_access_key()
    secret_key = generate_secret_key()
    full_key = f"{API_KEY_PREFIX}_{access_key}_{secret_key}"

    api_key = ApiKey(
        name=f"{app_name} - Auto Generated",
        description=f"Auto-generated API key for {app_name}",
        access_key=access_key,
        hashed_secret=hash_secret(secret_key),
    )
    db.add(api_key)
    await db.flush()

    return api_key, full_key


async def _find_available_port(db: AsyncSession, worker_id: int) -> int:
    """Find an available port on the worker."""
    result = await db.execute(
        select(App.port).where(App.worker_id == worker_id, App.port.isnot(None))
    )
    used_ports = {row[0] for row in result.fetchall()}

    port = 9000  # Start from 9000 to avoid conflicts with dev servers
    while port in used_ports:
        port += 1

    return port


async def _build_env_vars(
    request: Request,
    worker: Worker,
    app_def: dict,
    full_key: str,
    port: int,
    db: AsyncSession,
) -> dict:
    """Build environment variables for the app container."""
    lmstack_host = request.headers.get("host", "localhost:8000")
    lmstack_port = lmstack_host.split(":")[-1] if ":" in lmstack_host else "8000"

    host_ip = get_host_ip(request, worker)

    lmstack_api_url = f"http://{host_ip}:{lmstack_port}/v1"
    lmstack_base_url = f"http://{host_ip}:{lmstack_port}"
    app_url = f"http://{host_ip}:{port}"
    app_host_port = f"{host_ip}:{port}"
    app_secret_key = secrets.token_hex(32)

    # Get running models for {model_list} placeholder
    running_models_result = await db.execute(
        select(LLMModel.name)
        .join(Deployment, Deployment.model_id == LLMModel.id)
        .where(Deployment.status == "running")
    )
    running_model_names = [row[0] for row in running_models_result.fetchall()]
    model_list = (
        "-all," + ",".join(f"+{name}" for name in running_model_names)
        if running_model_names
        else ""
    )

    # Build env vars from template
    env_vars = {}
    for key, value in app_def["env_template"].items():
        if value == "{lmstack_api_url}":
            env_vars[key] = lmstack_api_url
        elif value == "{lmstack_base_url}":
            env_vars[key] = lmstack_base_url
        elif value == "{api_key}":
            env_vars[key] = full_key
        elif value == "{app_url}":
            env_vars[key] = app_url
        elif value == "{app_host_port}":
            env_vars[key] = app_host_port
        elif value == "{secret_key}":
            env_vars[key] = app_secret_key
        elif value == "{model_list}":
            env_vars[key] = model_list
        else:
            env_vars[key] = value

    return env_vars

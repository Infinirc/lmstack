"""Utility functions for Deploy Apps API.

Contains helper functions, constants, and common utilities used across
the apps module.
"""
import hashlib
import logging
import secrets
from typing import Optional

import httpx
from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.app import App, AppStatus
from app.models.worker import Worker
from app.schemas.app import AppResponse

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# HTTP client timeouts
DEFAULT_TIMEOUT = 30.0
CONTAINER_ACTION_TIMEOUT = 300.0  # 5 minutes for container operations
IMAGE_PULL_TIMEOUT = 600.0  # 10 minutes for large images

# API key prefix for generated keys
API_KEY_PREFIX = "lmsk"


# =============================================================================
# Cryptographic Utilities
# =============================================================================

def generate_access_key() -> str:
    """Generate a random access key (16 hex chars)."""
    return secrets.token_hex(8)


def generate_secret_key() -> str:
    """Generate a random secret key (32 hex chars)."""
    return secrets.token_hex(16)


def hash_secret(secret: str) -> str:
    """Hash a secret key for secure storage.

    Note: For production, consider using bcrypt or argon2 with salt.
    """
    return hashlib.sha256(secret.encode()).hexdigest()


# =============================================================================
# Database Helpers
# =============================================================================

async def get_worker_or_404(worker_id: int, db: AsyncSession) -> Worker:
    """Get worker by ID or raise 404 if not found or offline."""
    result = await db.execute(select(Worker).where(Worker.id == worker_id))
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

    if worker.status != "online":
        raise HTTPException(
            status_code=400,
            detail=f"Worker {worker.name} is not online (status: {worker.status})"
        )

    return worker


# =============================================================================
# Worker Communication
# =============================================================================

async def call_worker_api(
    worker: Worker,
    method: str,
    path: str,
    timeout: float = DEFAULT_TIMEOUT,
    **kwargs,
) -> dict:
    """Call worker agent API and handle errors.

    Args:
        worker: Worker to call
        method: HTTP method (GET, POST, DELETE, etc.)
        path: API path (e.g., "/containers")
        timeout: Request timeout in seconds
        **kwargs: Additional arguments passed to httpx.request()

    Returns:
        Response JSON as dict

    Raises:
        HTTPException: On worker errors
    """
    url = f"http://{worker.address}{path}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, **kwargs)

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Resource not found on worker")

            if response.status_code >= 400:
                detail = response.json().get("detail", response.text)
                raise HTTPException(status_code=response.status_code, detail=detail)

            return response.json()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to worker {worker.name} at {worker.address}"
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Worker {worker.name} request timed out"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error calling worker API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Response Conversion
# =============================================================================

def get_host_ip(request: Request, worker: Worker) -> str:
    """Determine the host IP that the container can use to reach LMStack.

    Args:
        request: FastAPI request object
        worker: Worker where app is deployed

    Returns:
        Host IP address string
    """
    import socket

    lmstack_host = request.headers.get("host", "localhost:8000")
    host_ip = lmstack_host.split(":")[0] if ":" in lmstack_host else lmstack_host

    # If host is localhost, try alternatives
    if host_ip in ("localhost", "127.0.0.1"):
        forwarded_host = request.headers.get("x-forwarded-host")
        if forwarded_host:
            host_ip = forwarded_host.split(":")[0]
        else:
            # Try to get our IP on the same network as the worker
            try:
                worker_ip = worker.address.split(":")[0]
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect((worker_ip, 80))
                host_ip = s.getsockname()[0]
                s.close()
            except (OSError, socket.error) as e:
                logger.warning(f"Could not determine host IP for worker {worker_ip}: {e}")
                host_ip = "host.docker.internal"  # Fallback for Docker Desktop

    return host_ip


def app_to_response(app: App, request: Request) -> AppResponse:
    """Convert App model to API response with computed access URL.

    Args:
        app: App database model (with worker relationship loaded)
        request: FastAPI request for building URLs

    Returns:
        AppResponse schema
    """
    worker_name = app.worker.name if app.worker else None
    worker_address = app.worker.address if app.worker else None

    # Build proxy URL (legacy)
    proxy_url = None
    if app.status == AppStatus.RUNNING.value and app.proxy_path:
        proxy_url = str(request.base_url).rstrip('/') + app.proxy_path

    # Build access URL based on proxy setting
    access_url = None
    if app.status == AppStatus.RUNNING.value and app.port:
        if app.use_proxy:
            # Use LMStack host with app port (nginx proxy)
            host = request.headers.get("host", "localhost:8000").split(":")[0]
            access_url = f"http://{host}:{app.port}"
        else:
            # Direct connection to worker
            if worker_address:
                worker_host = worker_address.split(":")[0]
                access_url = f"http://{worker_host}:{app.port}"

    return AppResponse(
        id=app.id,
        app_type=app.app_type,
        name=app.name,
        worker_id=app.worker_id,
        worker_name=worker_name,
        worker_address=worker_address,
        status=app.status,
        status_message=app.status_message,
        container_id=app.container_id,
        port=app.port,
        proxy_path=app.proxy_path,
        proxy_url=proxy_url,
        use_proxy=app.use_proxy,
        access_url=access_url,
        api_key_id=app.api_key_id,
        created_at=app.created_at,
        updated_at=app.updated_at,
    )

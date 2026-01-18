"""LMStack API Server"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.api import api_router
from app.api.app_proxy import router as app_proxy_router
from app.api.gateway import router as gateway_router
from app.config import get_settings
from app.core.exceptions import LMStackError
from app.database import async_session_maker, init_db
from app.models.worker import Worker, WorkerStatus
from app.services.app_sync import app_sync_service
from app.services.deployment_sync import deployment_sync_service
from app.services.worker_sync import worker_sync_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Background task control
_worker_check_task = None
_deployment_health_task = None
_app_health_task = None

# Health check interval (in seconds)
DEPLOYMENT_HEALTH_CHECK_INTERVAL = 60  # Check every minute
APP_HEALTH_CHECK_INTERVAL = 30  # Check apps more frequently


async def check_worker_status():
    """Background task to check worker heartbeat and mark offline workers."""
    while True:
        try:
            await asyncio.sleep(settings.worker_heartbeat_interval)

            async with async_session_maker() as db:
                # Find workers that haven't sent heartbeat within timeout
                timeout_threshold = datetime.now(UTC) - timedelta(seconds=settings.worker_timeout)

                # Get workers that are online but haven't sent heartbeat
                result = await db.execute(
                    select(Worker).where(
                        Worker.status == WorkerStatus.ONLINE.value,
                        Worker.last_heartbeat < timeout_threshold,
                    )
                )
                stale_workers = result.scalars().all()

                for worker in stale_workers:
                    # Check if it's a local worker (they don't send heartbeats)
                    labels = worker.labels or {}
                    if labels.get("type") == "local":
                        continue

                    logger.warning(
                        f"Worker {worker.name} (id={worker.id}) missed heartbeat, "
                        f"last seen: {worker.last_heartbeat}, marking as offline"
                    )
                    worker.status = WorkerStatus.OFFLINE.value

                await db.commit()

        except asyncio.CancelledError:
            logger.info("Worker status check task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in worker status check: {e}")
            await asyncio.sleep(10)  # Wait before retrying on error


async def check_deployment_health():
    """Background task to periodically check deployment health.

    This ensures that deployments marked as 'starting' eventually become
    'running' or 'error', and catches any containers that crash.
    """
    # Initial delay to let containers stabilize after startup sync
    await asyncio.sleep(30)

    while True:
        try:
            await asyncio.sleep(DEPLOYMENT_HEALTH_CHECK_INTERVAL)

            # Only sync deployments that are in transitional states
            stats = await deployment_sync_service.sync_all_deployments()

            if stats["total"] > 0:
                logger.debug(
                    f"Deployment health check: {stats['running_verified']} healthy, "
                    f"{stats['api_not_ready']} loading, {stats['container_missing']} missing"
                )

        except asyncio.CancelledError:
            logger.info("Deployment health check task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in deployment health check: {e}")
            await asyncio.sleep(30)  # Wait before retrying on error


async def check_app_health():
    """Background task to periodically check app container health.

    This ensures that apps marked as 'running' are actually running,
    and catches any containers that are manually deleted or crashed.
    """
    # Initial delay to let system stabilize
    await asyncio.sleep(15)

    while True:
        try:
            await asyncio.sleep(APP_HEALTH_CHECK_INTERVAL)

            stats = await app_sync_service.sync_all_apps()

            if stats["total"] > 0:
                logger.debug(
                    f"App health check: {stats['running_verified']} healthy, "
                    f"{stats['container_missing']} missing"
                )

        except asyncio.CancelledError:
            logger.info("App health check task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in app health check: {e}")
            await asyncio.sleep(30)  # Wait before retrying on error


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global _worker_check_task, _deployment_health_task, _app_health_task

    # Startup
    logger.info("Starting LMStack API Server...")
    await init_db()
    logger.info("Database initialized")

    # Check all workers' status first, then refresh resources on online workers
    try:
        logger.info("Checking worker status...")
        worker_stats = await worker_sync_service.sync_all_workers()
        if worker_stats["total"] > 0:
            logger.info(
                f"Worker sync complete: {worker_stats['online']} online, "
                f"{worker_stats['offline']} offline"
            )
        # Refresh resources on online workers
        if worker_stats["online"] > 0:
            await worker_sync_service.refresh_online_workers_resources()
    except Exception as e:
        logger.error(f"Failed to sync workers on startup: {e}")

    # Synchronize deployment status with actual container state
    # This is important after system reboot
    try:
        logger.info("Synchronizing deployment status...")
        sync_stats = await deployment_sync_service.sync_all_deployments()
        if sync_stats["total"] > 0:
            logger.info(
                f"Deployment sync complete: {sync_stats['running_verified']} running, "
                f"{sync_stats['restarting']} restarting, {sync_stats['api_not_ready']} loading, "
                f"{sync_stats['container_missing']} missing"
            )
    except Exception as e:
        logger.error(f"Failed to sync deployments on startup: {e}")

    # Synchronize app status with actual container state
    try:
        logger.info("Synchronizing app status...")
        app_stats = await app_sync_service.sync_all_apps()
        if app_stats["total"] > 0:
            logger.info(
                f"App sync complete: {app_stats['running_verified']} running, "
                f"{app_stats['container_missing']} missing"
            )
    except Exception as e:
        logger.error(f"Failed to sync apps on startup: {e}")

    # Start background task for checking worker status
    _worker_check_task = asyncio.create_task(check_worker_status())
    logger.info("Worker status check task started")

    # Start background task for checking deployment health
    _deployment_health_task = asyncio.create_task(check_deployment_health())
    logger.info("Deployment health check task started")

    # Start background task for checking app health
    _app_health_task = asyncio.create_task(check_app_health())
    logger.info("App health check task started")

    yield

    # Shutdown
    logger.info("Shutting down LMStack API Server...")

    # Cancel background tasks
    if _worker_check_task:
        _worker_check_task.cancel()
        try:
            await _worker_check_task
        except asyncio.CancelledError:
            pass
        logger.info("Worker status check task stopped")

    if _deployment_health_task:
        _deployment_health_task.cancel()
        try:
            await _deployment_health_task
        except asyncio.CancelledError:
            pass
        logger.info("Deployment health check task stopped")

    if _app_health_task:
        _app_health_task.cancel()
        try:
            await _app_health_task
        except asyncio.CancelledError:
            pass
        logger.info("App health check task stopped")


app = FastAPI(
    title=settings.app_name,
    description="LLM Deployment Management Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware configuration
# Note: allow_credentials=True with allow_origins=["*"] is insecure
# For production, set LMSTACK_CORS_ORIGINS to specific domains
cors_origins = settings.get_cors_origins()
allow_credentials = cors_origins != ["*"]  # Disable credentials for wildcard

if cors_origins == ["*"] and not settings.debug:
    logger.warning(
        "CORS is configured to allow all origins. "
        "For production, set LMSTACK_CORS_ORIGINS to specific domains."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers for custom exceptions
@app.exception_handler(LMStackError)
async def lmstack_exception_handler(request: Request, exc: LMStackError):
    """Handle custom LMStack exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


# Include API routes
app.include_router(api_router, prefix="/api")

# Include Gateway routes (OpenAI-compatible endpoints)
app.include_router(gateway_router, prefix="/v1", tags=["gateway"])

# Include App Proxy routes (for deployed apps like Open WebUI)
app.include_router(app_proxy_router, prefix="/apps", tags=["app-proxy"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "app": settings.app_name}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

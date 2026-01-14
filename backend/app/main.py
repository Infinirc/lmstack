"""LMStack API Server"""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, update

from app.api import api_router
from app.api.gateway import router as gateway_router
from app.api.app_proxy import router as app_proxy_router
from app.config import get_settings
from app.core.exceptions import LMStackError
from app.database import init_db, async_session_maker
from app.models.worker import Worker, WorkerStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Background task control
_worker_check_task = None


async def check_worker_status():
    """Background task to check worker heartbeat and mark offline workers."""
    while True:
        try:
            await asyncio.sleep(settings.worker_heartbeat_interval)

            async with async_session_maker() as db:
                # Find workers that haven't sent heartbeat within timeout
                timeout_threshold = datetime.now(timezone.utc) - timedelta(seconds=settings.worker_timeout)

                # Get workers that are online but haven't sent heartbeat
                result = await db.execute(
                    select(Worker).where(
                        Worker.status == WorkerStatus.ONLINE.value,
                        Worker.last_heartbeat < timeout_threshold
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global _worker_check_task

    # Startup
    logger.info("Starting LMStack API Server...")
    await init_db()
    logger.info("Database initialized")

    # Start background task for checking worker status
    _worker_check_task = asyncio.create_task(check_worker_status())
    logger.info("Worker status check task started")

    yield

    # Shutdown
    logger.info("Shutting down LMStack API Server...")

    # Cancel background task
    if _worker_check_task:
        _worker_check_task.cancel()
        try:
            await _worker_check_task
        except asyncio.CancelledError:
            pass
        logger.info("Worker status check task stopped")


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

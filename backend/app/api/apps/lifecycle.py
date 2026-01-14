"""App lifecycle management endpoints.

Contains endpoints for starting, stopping, and managing app state.
"""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.app import App, AppStatus
from app.schemas.app import AppResponse, AppLogsResponse
from app.services.app_proxy_manager import get_proxy_manager
from app.api.apps.utils import (
    DEFAULT_TIMEOUT,
    CONTAINER_ACTION_TIMEOUT,
    call_worker_api,
    app_to_response,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/{app_id}/stop", response_model=AppResponse)
async def stop_app(
    app_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Stop a running app."""
    result = await db.execute(select(App).where(App.id == app_id))
    app = result.scalar_one_or_none()

    if not app:
        raise HTTPException(status_code=404, detail="App not found")

    if app.status != AppStatus.RUNNING.value:
        raise HTTPException(status_code=400, detail="App is not running")

    await db.refresh(app, ["worker"])
    worker = app.worker

    if not worker or worker.status != "online":
        raise HTTPException(status_code=400, detail="Worker is not available")

    try:
        app.status = AppStatus.STOPPING.value
        await db.commit()

        await call_worker_api(
            worker,
            "POST",
            f"/containers/{app.container_id}/stop",
            params={"timeout": 10},
            timeout=CONTAINER_ACTION_TIMEOUT,
        )

        app.status = AppStatus.STOPPED.value
        await db.commit()

    except Exception as e:
        logger.exception(f"Failed to stop app: {e}")
        app.status = AppStatus.ERROR.value
        app.status_message = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to stop app: {e}")

    return app_to_response(app, request)


@router.post("/{app_id}/start", response_model=AppResponse)
async def start_app(
    app_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Start a stopped app."""
    result = await db.execute(select(App).where(App.id == app_id))
    app = result.scalar_one_or_none()

    if not app:
        raise HTTPException(status_code=404, detail="App not found")

    if app.status not in [AppStatus.STOPPED.value, AppStatus.ERROR.value]:
        raise HTTPException(status_code=400, detail="App is not stopped")

    await db.refresh(app, ["worker"])
    worker = app.worker

    if not worker or worker.status != "online":
        raise HTTPException(status_code=400, detail="Worker is not available")

    try:
        app.status = AppStatus.STARTING.value
        await db.commit()

        await call_worker_api(
            worker,
            "POST",
            f"/containers/{app.container_id}/start",
            timeout=CONTAINER_ACTION_TIMEOUT,
        )

        app.status = AppStatus.RUNNING.value
        app.status_message = None
        await db.commit()

    except Exception as e:
        logger.exception(f"Failed to start app: {e}")
        app.status = AppStatus.ERROR.value
        app.status_message = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to start app: {e}")

    return app_to_response(app, request)


@router.delete("/{app_id}", status_code=204)
async def delete_app(
    app_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete an app and its container."""
    result = await db.execute(select(App).where(App.id == app_id))
    app = result.scalar_one_or_none()

    if not app:
        raise HTTPException(status_code=404, detail="App not found")

    await db.refresh(app, ["worker", "api_key"])

    # Try to remove container if it exists
    if app.container_id and app.worker and app.worker.status == "online":
        try:
            await call_worker_api(
                app.worker,
                "DELETE",
                f"/containers/{app.container_id}",
                params={"force": True, "volumes": False},
            )
        except Exception as e:
            logger.warning(f"Failed to remove container: {e}")

    # Remove nginx proxy (if it was enabled)
    if app.use_proxy:
        try:
            proxy_manager = get_proxy_manager()
            await proxy_manager.remove_app_proxy(app_id)
        except Exception as e:
            logger.warning(f"Failed to remove nginx proxy: {e}")

    # Delete associated API key
    if app.api_key:
        await db.delete(app.api_key)

    # Delete app record
    await db.delete(app)
    await db.commit()


@router.get("/{app_id}/logs", response_model=AppLogsResponse)
async def get_app_logs(
    app_id: int,
    tail: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    """Get logs for an app container."""
    result = await db.execute(select(App).where(App.id == app_id))
    app = result.scalar_one_or_none()

    if not app:
        raise HTTPException(status_code=404, detail="App not found")

    if not app.container_id:
        return AppLogsResponse(app_id=app_id, logs="No container running")

    await db.refresh(app, ["worker"])
    worker = app.worker

    if not worker or worker.status != "online":
        return AppLogsResponse(app_id=app_id, logs="Worker is not available")

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                f"http://{worker.address}/containers/{app.container_id}/logs",
                params={"tail": tail},
            )
            if response.status_code >= 400:
                return AppLogsResponse(app_id=app_id, logs=f"Failed to get logs: {response.text}")

            logs_data = response.json()
            logs = logs_data.get("logs", "")
    except Exception as e:
        logger.exception(f"Failed to get logs for app {app_id}: {e}")
        return AppLogsResponse(app_id=app_id, logs=f"Failed to get logs: {e}")

    return AppLogsResponse(app_id=app_id, logs=logs)

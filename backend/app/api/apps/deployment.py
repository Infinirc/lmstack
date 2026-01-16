"""Background deployment logic for Deploy Apps.

Contains the background task logic for deploying apps including:
- Image pulling with progress tracking
- Container creation and health checking
- Nginx proxy setup
"""

import asyncio
import logging

import httpx
from sqlalchemy import select

from app.api.apps.utils import CONTAINER_ACTION_TIMEOUT, DEFAULT_TIMEOUT, IMAGE_PULL_TIMEOUT
from app.models.app import App, AppStatus, AppType
from app.models.worker import Worker
from app.services.app_proxy_manager import get_proxy_manager

logger = logging.getLogger(__name__)


# =============================================================================
# Progress Tracking
# =============================================================================

# In-memory store for tracking deployment progress.
# NOTE: This is not suitable for multi-process deployments. For production
# with multiple workers, use Redis or store progress in the database.
_deployment_progress: dict[int, dict] = {}
_MAX_PROGRESS_ENTRIES = 100  # Limit to prevent memory leaks


def get_deployment_progress(app_id: int) -> dict:
    """Get deployment progress for an app."""
    if app_id in _deployment_progress:
        return _deployment_progress[app_id]
    return {"stage": "unknown", "progress": 0, "message": "No progress data"}


def set_deployment_progress(app_id: int, stage: str, progress: int, message: str) -> None:
    """Set deployment progress for an app."""
    _deployment_progress[app_id] = {
        "stage": stage,
        "progress": progress,
        "message": message,
    }


def cleanup_old_progress_entries() -> None:
    """Remove old progress entries to prevent memory leaks."""
    if len(_deployment_progress) > _MAX_PROGRESS_ENTRIES:
        # Remove completed or errored entries first
        to_remove = [
            app_id
            for app_id, progress in _deployment_progress.items()
            if progress.get("stage") in ("completed", "error", "running")
        ]
        for app_id in to_remove[: len(_deployment_progress) - _MAX_PROGRESS_ENTRIES // 2]:
            _deployment_progress.pop(app_id, None)


# =============================================================================
# Image Pulling
# =============================================================================


async def pull_image_with_progress(
    worker: Worker,
    image: str,
    app_id: int,
) -> bool:
    """Pull a Docker image with progress tracking.

    Args:
        worker: Worker to pull image on
        image: Docker image name
        app_id: App ID for progress tracking

    Returns:
        True if successful

    Raises:
        Exception: On pull failure
    """
    url = f"http://{worker.address}/images/pull"
    progress_url = f"http://{worker.address}/images/pull-progress/{app_id}"

    set_deployment_progress(app_id, "pulling", 0, f"Pulling image {image}...")

    try:
        async with httpx.AsyncClient(timeout=IMAGE_PULL_TIMEOUT) as client:
            # Start the pull request in a task with app_id for progress tracking
            pull_task = asyncio.create_task(
                client.post(url, json={"image": image, "app_id": app_id})
            )

            # Poll for progress while waiting
            while not pull_task.done():
                try:
                    progress_resp = await client.get(progress_url, timeout=5.0)
                    if progress_resp.status_code == 200:
                        progress_data = progress_resp.json()
                        status = progress_data.get("status", "")
                        progress = progress_data.get("progress", 0)

                        if status == "pulling":
                            set_deployment_progress(
                                app_id,
                                "pulling",
                                progress,
                                f"Pulling image {image}... ({progress}%)",
                            )
                        elif status == "completed":
                            set_deployment_progress(
                                app_id,
                                "pulling",
                                100,
                                "Image pulled successfully",
                            )
                except Exception:
                    pass  # Progress polling is best-effort

                await asyncio.sleep(2)

            # Get the final response
            response = await pull_task
            if response.status_code >= 400:
                raise Exception(f"Failed to pull image: {response.text}")

            set_deployment_progress(app_id, "pulling", 100, "Image pulled successfully")
            return True
    except Exception as e:
        set_deployment_progress(app_id, "error", 0, str(e))
        raise


# =============================================================================
# Container Health Checking
# =============================================================================


async def wait_for_container_healthy(
    worker_address: str,
    container_id: str,
    app_id: int,
    port: int,
    poll_interval: int = 2,
) -> bool:
    """Wait for container to become healthy.

    Args:
        worker_address: Worker address (host:port)
        container_id: Container ID to check
        app_id: App ID for progress tracking
        port: App port for HTTP health check
        poll_interval: Time between checks in seconds

    Returns:
        True if healthy (waits indefinitely until healthy or error)
    """
    waited = 0
    consecutive_failures = 0
    max_consecutive_failures = 10  # Fail after 20 seconds of no connection
    slow_threshold = 1800  # 30 minutes before showing "check" message
    shown_slow_message = False

    worker_host = worker_address.split(":")[0]
    app_url = f"http://{worker_host}:{port}"

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        while True:  # Wait indefinitely
            try:
                response = await client.get(f"http://{worker_address}/containers/{container_id}")

                # Reset failure counter on successful connection
                consecutive_failures = 0

                # Container was deleted externally
                if response.status_code == 404:
                    raise Exception("Container was deleted or not found")

                if response.status_code == 200:
                    container_info = response.json()
                    state = container_info.get("state", "").lower()
                    status = container_info.get("status", "").lower()

                    logger.debug(f"App {app_id} container state={state}, status={status}")

                    if state == "running":
                        # Check health status in the status string
                        if "unhealthy" in status:
                            raise Exception("Container is unhealthy")

                        elif "healthy)" in status:
                            # Container reports healthy, verify HTTP access
                            set_deployment_progress(
                                app_id,
                                "starting",
                                95,
                                "Almost ready, verifying accessibility...",
                            )
                            if await _verify_http_access(client, app_url, app_id):
                                return True

                        elif "health:" in status or "starting)" in status:
                            # Health check still running
                            mins = waited // 60
                            secs = waited % 60
                            time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

                            if waited >= slow_threshold and not shown_slow_message:
                                set_deployment_progress(
                                    app_id,
                                    "starting",
                                    80,
                                    f"App is initializing ({time_str}) - Taking longer than expected. "
                                    "Please check container logs for issues.",
                                )
                                shown_slow_message = True
                            elif shown_slow_message:
                                set_deployment_progress(
                                    app_id,
                                    "starting",
                                    80,
                                    f"App is initializing ({time_str}) - Please check logs if needed.",
                                )
                            else:
                                progress_pct = min(50 + int(waited / 600 * 40), 90)
                                set_deployment_progress(
                                    app_id,
                                    "starting",
                                    progress_pct,
                                    f"App is initializing ({time_str}, please wait)...",
                                )

                        elif "health" not in status:
                            # No health check defined, verify HTTP access directly
                            if waited >= 10:
                                set_deployment_progress(
                                    app_id,
                                    "starting",
                                    90,
                                    "Almost ready, verifying accessibility...",
                                )
                                if await _verify_http_access(client, app_url, app_id):
                                    return True
                            else:
                                set_deployment_progress(
                                    app_id, "starting", 70, "Container starting..."
                                )

                    elif state in ["exited", "dead"]:
                        raise Exception(f"Container stopped unexpectedly: {status}")

            except httpx.RequestError as e:
                consecutive_failures += 1
                logger.warning(
                    f"Failed to check container status (attempt {consecutive_failures}): {e}"
                )
                if consecutive_failures >= max_consecutive_failures:
                    raise Exception(f"Worker unreachable after {consecutive_failures} attempts")

            await asyncio.sleep(poll_interval)
            waited += poll_interval


async def _verify_http_access(
    client: httpx.AsyncClient,
    app_url: str,
    app_id: int,
) -> bool:
    """Verify app is accessible via HTTP."""
    try:
        http_check = await client.get(app_url, timeout=10.0)
        if http_check.status_code < 500:
            logger.info(f"App {app_id} HTTP check passed: {http_check.status_code}")
            return True
        else:
            logger.warning(f"App {app_id} HTTP check failed: {http_check.status_code}")
    except Exception as http_err:
        logger.warning(f"App {app_id} HTTP check error: {http_err}")
    return False


# =============================================================================
# Main Deployment Task
# =============================================================================


async def deploy_app_background(
    app_id: int,
    app_type: AppType,
    worker_address: str,
    worker_host: str,
    env_vars: dict,
    port: int,
    app_def: dict,
    lmstack_port: str,
    use_proxy: bool = True,
) -> None:
    """Background task to deploy an app.

    This function handles the complete deployment lifecycle:
    1. Pull Docker image
    2. Create and start container
    3. Wait for health check
    4. Setup nginx proxy (if enabled)

    Args:
        app_id: App database ID
        app_type: Type of app being deployed
        worker_address: Worker address (host:port)
        worker_host: Worker hostname
        env_vars: Environment variables for container
        port: Host port for app
        app_def: App definition from APP_DEFINITIONS
        lmstack_port: LMStack API port
        use_proxy: Whether to setup nginx proxy
    """
    from app.database import async_session_maker

    # Cleanup old progress entries to prevent memory leaks
    cleanup_old_progress_entries()

    async with async_session_maker() as db:
        try:
            # Get app from database
            result = await db.execute(select(App).where(App.id == app_id))
            app = result.scalar_one_or_none()
            if not app:
                logger.error(f"App {app_id} not found")
                return

            # Get worker
            result = await db.execute(select(Worker).where(Worker.id == app.worker_id))
            worker = result.scalar_one_or_none()
            if not worker:
                logger.error(f"Worker not found for app {app_id}")
                app.status = AppStatus.ERROR.value
                app.status_message = "Worker not found"
                await db.commit()
                return

            # Phase 1: Pull image
            app.status = AppStatus.PULLING.value
            app.status_message = "Pulling image..."
            await db.commit()

            try:
                await pull_image_with_progress(worker, app_def["image"], app_id)
            except Exception as e:
                app.status = AppStatus.ERROR.value
                app.status_message = f"Failed to pull image: {e}"
                await db.commit()
                return

            # Phase 2: Create container
            app.status = AppStatus.STARTING.value
            app.status_message = "Starting container..."
            await db.commit()

            set_deployment_progress(app_id, "starting", 0, "Creating container...")

            container_id = await _create_container(
                worker_address, app_id, app_type, app_def, env_vars, port
            )
            if not container_id:
                return

            # Update app with container info
            app.container_id = container_id
            app.port = port
            await db.commit()

            # Phase 3: Wait for health (waits indefinitely until healthy or error)
            set_deployment_progress(
                app_id,
                "starting",
                50,
                "Waiting for app to start...",
            )

            await wait_for_container_healthy(
                worker_address=worker_address,
                container_id=container_id,
                app_id=app_id,
                port=port,
            )

            # Phase 4: Setup proxy
            if use_proxy:
                await _setup_nginx_proxy(app_id, app_type, worker_address, port)
            else:
                logger.info(f"Proxy disabled for app {app_id}, using direct worker connection")

            # Mark as running
            app.status = AppStatus.RUNNING.value
            app.status_message = None
            await db.commit()

            set_deployment_progress(app_id, "running", 100, "App deployed successfully")
            logger.info(f"App {app_id} deployed successfully")

        except Exception as e:
            logger.exception(f"Failed to deploy app {app_id}: {e}")
            try:
                result = await db.execute(select(App).where(App.id == app_id))
                app = result.scalar_one_or_none()
                if app:
                    app.status = AppStatus.ERROR.value
                    app.status_message = str(e)
                    await db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update app error status: {db_error}")

            set_deployment_progress(app_id, "error", 0, str(e))


async def _create_container(
    worker_address: str,
    app_id: int,
    app_type: AppType,
    app_def: dict,
    env_vars: dict,
    port: int,
) -> str | None:
    """Create container on worker.

    Returns:
        Container ID if successful, None on failure
    """
    from app.database import async_session_maker

    container_name = f"lmstack-app-{app_type.value}"

    # Build volumes
    volumes = []
    for vol in app_def.get("volumes", []):
        volumes.append(
            {
                "source": f"{container_name}-{vol['name']}",
                "destination": vol["destination"],
                "mode": "rw",
            }
        )

    payload = {
        "name": container_name,
        "image": app_def["image"],
        "env": env_vars,
        "ports": [
            {
                "container_port": app_def["internal_port"],
                "host_port": port,
                "protocol": "tcp",
            }
        ],
        "volumes": volumes,
        "restart_policy": "unless-stopped",
        "labels": {
            "lmstack.app": "true",
            "lmstack.app.type": app_type.value,
            "lmstack.app.id": str(app_id),
        },
    }

    # Add Linux capabilities if specified (e.g., SYS_ADMIN for AnythingLLM)
    if app_def.get("cap_add"):
        payload["cap_add"] = app_def["cap_add"]

    try:
        async with httpx.AsyncClient(timeout=CONTAINER_ACTION_TIMEOUT) as client:
            response = await client.post(
                f"http://{worker_address}/containers",
                json=payload,
            )
            if response.status_code >= 400:
                raise Exception(f"Failed to create container: {response.text}")

            container_data = response.json()
            return container_data.get("id")

    except Exception as e:
        async with async_session_maker() as db:
            result = await db.execute(select(App).where(App.id == app_id))
            app = result.scalar_one_or_none()
            if app:
                app.status = AppStatus.ERROR.value
                app.status_message = f"Failed to create container: {e}"
                await db.commit()

        set_deployment_progress(app_id, "error", 0, str(e))
        return None


async def _setup_nginx_proxy(
    app_id: int,
    app_type: AppType,
    worker_address: str,
    port: int,
) -> None:
    """Setup nginx proxy for app."""
    set_deployment_progress(app_id, "starting", 95, "Setting up proxy...")

    try:
        proxy_manager = get_proxy_manager()
        proxy_worker_host = worker_address.split(":")[0]
        await proxy_manager.add_app_proxy(
            app_id=app_id,
            app_type=app_type.value,
            listen_port=port,
            worker_host=proxy_worker_host,
            worker_port=port,
        )
        logger.info(f"Nginx proxy configured for app {app_id}")
    except Exception as e:
        logger.warning(f"Failed to setup nginx proxy: {e}")
        # Continue anyway, user can access directly via worker IP

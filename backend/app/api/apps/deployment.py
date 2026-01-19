"""Background deployment logic for Deploy Apps.

Contains the background task logic for deploying apps including:
- Image pulling with progress tracking
- Container creation and health checking
- Nginx proxy setup
- Semantic Router config generation
"""

import asyncio
import logging

import httpx
from sqlalchemy import select

from app.api.apps.utils import CONTAINER_ACTION_TIMEOUT, DEFAULT_TIMEOUT, IMAGE_PULL_TIMEOUT
from app.models.app import App, AppStatus, AppType
from app.models.worker import Worker
from app.services.app_proxy_manager import get_proxy_manager
from app.services.semantic_router import semantic_router_service

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
            # Track last known progress to avoid regression when status is "unknown"
            last_known_progress = 0

            while not pull_task.done():
                try:
                    progress_resp = await client.get(progress_url, timeout=5.0)
                    if progress_resp.status_code == 200:
                        progress_data = progress_resp.json()
                        status = progress_data.get("status", "")
                        progress = progress_data.get("progress", 0)

                        if status == "pulling":
                            # Only update if progress is moving forward (avoid regression)
                            if progress >= last_known_progress:
                                last_known_progress = progress
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
                        # Ignore "unknown" status - keep showing last known progress
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
    """Verify app is accessible via HTTP.

    Returns True if the app responds to HTTP requests (any status code).
    A 500 error still means the app is running and accepting connections,
    just that it may be initializing or the endpoint doesn't exist.
    """
    try:
        http_check = await client.get(app_url, timeout=10.0)
        # Any HTTP response (including 500) means the app is running
        # Only connection errors should be treated as failures
        logger.info(f"App {app_id} HTTP check passed: {http_check.status_code}")
        return True
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
    lmstack_host: str | None = None,
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
        lmstack_host: LMStack API host (for semantic router config)
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

            # Phase 2: Pre-deployment setup (e.g., config files)
            if app_type == AppType.SEMANTIC_ROUTER:
                set_deployment_progress(app_id, "starting", 0, "Generating config...")
                try:
                    # Use lmstack_host parameter, or fallback to LMSTACK_BACKEND_URL env var
                    import os

                    if lmstack_host:
                        lmstack_api_url = f"http://{lmstack_host}:{lmstack_port}"
                    else:
                        backend_url = os.environ.get("LMSTACK_BACKEND_URL")
                        if backend_url:
                            lmstack_api_url = backend_url.rstrip("/")
                        else:
                            # Last resort: use worker host (may not work from container)
                            lmstack_api_url = f"http://{worker_host}:{lmstack_port}"
                    logger.info(f"Semantic router will use LMStack API: {lmstack_api_url}")
                    # Get API key from app config
                    api_key = (app.config or {}).get("api_key")
                    await write_semantic_router_config(worker_address, lmstack_api_url, db, api_key)
                except Exception as e:
                    logger.warning(f"Failed to write semantic router config: {e}")
                    # Continue anyway, config can be updated later

            # Phase 3: Create container
            app.status = AppStatus.STARTING.value
            app.status_message = "Starting container..."
            await db.commit()

            set_deployment_progress(app_id, "starting", 10, "Creating container...")

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
                await _setup_nginx_proxy(app_id, app_type, worker_address, port, app_def)
            else:
                logger.info(f"Proxy disabled for app {app_id}, using direct worker connection")

            # Mark as running
            app.status = AppStatus.RUNNING.value
            app.status_message = None
            await db.commit()

            set_deployment_progress(app_id, "running", 100, "App deployed successfully")
            logger.info(f"App {app_id} deployed successfully")

            # Auto-deploy monitoring services for apps that support it
            if app_def.get("has_monitoring"):
                logger.info(f"Auto-deploying monitoring services for app {app_id}")
                await _auto_deploy_monitoring(db, app, worker, port)

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

    # Build port mappings
    ports = [
        {
            "container_port": app_def["internal_port"],
            "host_port": port,
            "protocol": "tcp",
        }
    ]

    # Add additional ports (e.g., dashboard port for semantic router)
    additional_ports = app_def.get("additional_ports", [])
    for i, additional_port_info in enumerate(additional_ports):
        # Handle both old format (int) and new format (dict with container_port and name)
        if isinstance(additional_port_info, dict):
            container_port = additional_port_info["container_port"]
        else:
            container_port = additional_port_info
        # Map additional ports starting from port + 1
        host_port = port + 1 + i
        ports.append(
            {
                "container_port": container_port,
                "host_port": host_port,
                "protocol": "tcp",
            }
        )

    payload = {
        "name": container_name,
        "image": app_def["image"],
        "env": env_vars,
        "ports": ports,
        "volumes": volumes,
        "restart_policy": "unless-stopped",
        "labels": {
            "lmstack.app": "true",
            "lmstack.app.type": app_type.value,
            "lmstack.app.id": str(app_id),
        },
        # Add host.docker.internal mapping for container to access host services
        "extra_hosts": {"host.docker.internal": "host-gateway"},
    }

    # Add entrypoint/command if specified (e.g., for semantic router config symlink)
    if app_def.get("entrypoint"):
        payload["entrypoint"] = app_def["entrypoint"]
    if app_def.get("command"):
        payload["command"] = app_def["command"]

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
    app_def: dict | None = None,
) -> None:
    """Setup nginx proxy for app and its additional ports."""
    set_deployment_progress(app_id, "starting", 95, "Setting up proxy...")

    try:
        proxy_manager = get_proxy_manager()
        proxy_worker_host = worker_address.split(":")[0]

        # Setup main port proxy
        await proxy_manager.add_app_proxy(
            app_id=app_id,
            app_type=app_type.value,
            listen_port=port,
            worker_host=proxy_worker_host,
            worker_port=port,
        )
        logger.info(f"Nginx proxy configured for app {app_id} main port {port}")

        # Setup additional port proxies (e.g., dashboard for semantic router)
        if app_def:
            additional_ports = app_def.get("additional_ports", [])
            for i, port_info in enumerate(additional_ports):
                if isinstance(port_info, dict):
                    port_name = port_info.get("name", f"port{i+1}")
                else:
                    port_name = f"port{i+1}"

                host_port = port + 1 + i
                await proxy_manager.add_app_proxy(
                    app_id=app_id * 1000 + i + 1,  # Unique ID for additional port
                    app_type=f"{app_type.value}-{port_name.lower()}",
                    listen_port=host_port,
                    worker_host=proxy_worker_host,
                    worker_port=host_port,
                )
                logger.info(f"Nginx proxy configured for app {app_id} {port_name} port {host_port}")

    except Exception as e:
        logger.warning(f"Failed to setup nginx proxy: {e}")
        # Continue anyway, user can access directly via worker IP


# =============================================================================
# Semantic Router Config
# =============================================================================


async def write_semantic_router_config(
    worker_address: str,
    lmstack_api_url: str,
    db,
    api_key: str | None = None,
) -> None:
    """Write semantic router config.yaml to the worker volume.

    Args:
        worker_address: Worker address (host:port)
        lmstack_api_url: LMStack API URL for the semantic router to call
        db: Database session
        api_key: LMStack API key for authentication
    """
    # Generate config with API key
    config = await semantic_router_service.generate_config(db, lmstack_api_url, api_key)
    config_yaml = semantic_router_service.config_to_yaml(config)

    # Write to worker volume
    volume_name = "lmstack-app-semantic-router-semantic-router-config"

    async with httpx.AsyncClient(timeout=CONTAINER_ACTION_TIMEOUT) as client:
        response = await client.post(
            f"http://{worker_address}/storage/volumes/write-file",
            json={
                "volume_name": volume_name,
                "file_path": "config.yaml",
                "content": config_yaml,
            },
        )
        if response.status_code >= 400:
            raise Exception(f"Failed to write config: {response.text}")

    logger.info(f"Wrote semantic router config to {volume_name}/config.yaml")


async def update_semantic_router_config_if_deployed(db) -> bool:
    """Update semantic router config if it's deployed.

    This should be called when deployments change (add/remove models).

    Args:
        db: Database session

    Returns:
        True if config was updated, False if semantic router not deployed
    """
    import os

    # Check if semantic router is deployed
    app = await semantic_router_service.get_semantic_router_app(db)
    if not app:
        return False

    # Get worker address
    result = await db.execute(select(Worker).where(Worker.id == app.worker_id))
    worker = result.scalar_one_or_none()
    if not worker:
        return False

    # Build LMStack API URL
    # Priority: 1) stored lmstack_host, 2) LMSTACK_BACKEND_URL env var, 3) worker IP
    app_config = app.config or {}
    lmstack_host = app_config.get("lmstack_host")

    if lmstack_host:
        lmstack_api_url = f"http://{lmstack_host}:52000"
    else:
        backend_url = os.environ.get("LMSTACK_BACKEND_URL")
        if backend_url:
            lmstack_api_url = backend_url.rstrip("/")
        else:
            # Fallback: use worker IP (may not work from container)
            worker_host = worker.address.split(":")[0]
            lmstack_api_url = f"http://{worker_host}:52000"

    logger.info(f"Updating semantic router config with LMStack API: {lmstack_api_url}")

    # Get API key from app config
    api_key = app_config.get("api_key")

    try:
        await write_semantic_router_config(worker.address, lmstack_api_url, db, api_key)
        logger.info("Updated semantic router config with latest deployments")

        # Restart envoy to apply new config
        if app.container_id:
            await _restart_semantic_router_envoy(worker.address, app.container_id)

        return True
    except Exception as e:
        logger.error(f"Failed to update semantic router config: {e}")
        return False


async def _restart_semantic_router_envoy(worker_address: str, container_id: str) -> None:
    """Restart envoy process inside semantic router container to apply new config.

    Args:
        worker_address: Worker address (host:port)
        container_id: Semantic router container ID
    """
    try:
        async with httpx.AsyncClient(timeout=CONTAINER_ACTION_TIMEOUT) as client:
            # Execute supervisorctl restart envoy inside the container
            response = await client.post(
                f"http://{worker_address}/containers/{container_id}/exec",
                json={
                    "command": ["supervisorctl", "restart", "envoy"],
                },
            )
            if response.status_code >= 400:
                logger.warning(f"Failed to restart envoy: {response.text}")
            else:
                logger.info("Restarted semantic router envoy to apply new config")
    except Exception as e:
        logger.warning(f"Failed to restart semantic router envoy: {e}")
        # Don't raise - config was updated, envoy restart is best-effort


# =============================================================================
# Auto-deploy Monitoring
# =============================================================================


async def _auto_deploy_monitoring(
    db,
    parent_app: App,
    worker: Worker,
    parent_port: int,
) -> None:
    """Auto-deploy monitoring services (Grafana, Prometheus, Jaeger) for apps that support it.

    This is called automatically after Semantic Router deployment completes.
    Deploys services sequentially: Prometheus first (Grafana needs it), then Grafana, then Jaeger.
    """
    from app.api.apps.monitoring import deploy_monitoring_background
    from app.models.app import MONITORING_DEFINITIONS

    logger.info(f"Starting auto-deployment of monitoring services for app {parent_app.id}")

    # Find available ports starting from parent app's port + 10
    base_port = parent_port + 10
    result = await db.execute(
        select(App.port).where(App.worker_id == worker.id, App.port.isnot(None))
    )
    used_ports = {row[0] for row in result.fetchall()}

    # Create monitoring app records
    services_to_deploy = ["prometheus", "grafana", "jaeger"]
    created_apps = []
    port = base_port

    for svc_type in services_to_deploy:
        # Find next available port
        while port in used_ports:
            port += 1

        svc_def = MONITORING_DEFINITIONS[svc_type]
        svc_app = App(
            app_type=svc_type,
            name=f"{svc_def['name']} ({parent_app.name})",
            worker_id=worker.id,
            parent_app_id=parent_app.id,
            status=AppStatus.PENDING.value,
            proxy_path=f"/apps/{parent_app.app_type}/monitoring/{svc_type}",
            port=port,
            use_proxy=parent_app.use_proxy,
        )
        db.add(svc_app)
        await db.flush()
        created_apps.append((svc_app, svc_def))
        used_ports.add(port)
        port += 1

    await db.commit()

    # Find prometheus port for Grafana configuration
    prometheus_port = None
    for svc_app, _ in created_apps:
        if svc_app.app_type == "prometheus":
            prometheus_port = svc_app.port
            break

    # Deploy services sequentially (not in parallel) to ensure proper ordering
    # Prometheus must be ready before Grafana tries to configure its datasource
    for svc_app, svc_def in created_apps:
        logger.info(f"Deploying monitoring service: {svc_app.app_type}")
        try:
            await deploy_monitoring_background(
                app_id=svc_app.id,
                parent_app_id=parent_app.id,
                svc_type=svc_app.app_type,
                worker_address=worker.address,
                port=svc_app.port,
                svc_def=svc_def,
                use_proxy=svc_app.use_proxy,
                parent_app_port=parent_port,
                prometheus_port=prometheus_port,
            )
        except Exception as e:
            logger.error(f"Failed to deploy monitoring service {svc_app.app_type}: {e}")
            # Continue with other services even if one fails

    logger.info(f"Completed auto-deployment of monitoring services for app {parent_app.id}")

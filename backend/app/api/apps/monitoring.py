"""API routes for app monitoring services.

Manages Grafana, Prometheus, and Jaeger as sub-services of apps like Semantic Router.
"""

import asyncio
import base64
import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.apps.deployment import (
    pull_image_with_progress,
    set_deployment_progress,
    wait_for_container_healthy,
)
from app.api.apps.utils import CONTAINER_ACTION_TIMEOUT
from app.core.deps import require_operator, require_viewer
from app.database import get_db
from app.models.app import APP_DEFINITIONS, MONITORING_DEFINITIONS, App, AppStatus, AppType
from app.models.user import User
from app.models.worker import Worker
from app.services.app_proxy_manager import get_proxy_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Schemas
# =============================================================================


class MonitoringServiceStatus(BaseModel):
    """Status of a single monitoring service."""

    name: str
    type: str  # grafana, prometheus, jaeger
    status: str
    port: int | None = None
    url: str | None = None


class MonitoringStatus(BaseModel):
    """Overall monitoring status for an app."""

    enabled: bool
    services: list[MonitoringServiceStatus]


class MonitoringDeployRequest(BaseModel):
    """Request to deploy monitoring services."""

    services: list[str] | None = None  # ["grafana", "prometheus", "jaeger"], None = all


# =============================================================================
# Routes
# =============================================================================


@router.get("/{app_id}/monitoring", response_model=MonitoringStatus)
async def get_monitoring_status(
    app_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Get monitoring status for an app."""
    # Get parent app
    result = await db.execute(select(App).where(App.id == app_id))
    app = result.scalar_one_or_none()
    if not app:
        raise HTTPException(status_code=404, detail="App not found")

    # Check if app supports monitoring
    try:
        app_type = AppType(app.app_type)
        app_def = APP_DEFINITIONS.get(app_type, {})
        if not app_def.get("has_monitoring"):
            raise HTTPException(status_code=400, detail="This app does not support monitoring")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid app type")

    # Get child monitoring apps
    result = await db.execute(select(App).where(App.parent_app_id == app_id))
    child_apps = result.scalars().all()

    # Get worker for URL building
    await db.refresh(app, ["worker"])
    worker_host = app.worker.address.split(":")[0]

    # Use browser hostname if available
    host = request.headers.get("host", "").split(":")[0]
    if not host or host in ("localhost", "127.0.0.1"):
        host = worker_host

    services = []
    for child in child_apps:
        url = (
            f"http://{host}:{child.port}"
            if child.port and child.status == AppStatus.RUNNING.value
            else None
        )
        services.append(
            MonitoringServiceStatus(
                name=MONITORING_DEFINITIONS.get(child.app_type, {}).get("name", child.app_type),
                type=child.app_type,
                status=child.status,
                port=child.port,
                url=url,
            )
        )

    return MonitoringStatus(
        enabled=len(services) > 0,
        services=services,
    )


@router.post("/{app_id}/monitoring", response_model=MonitoringStatus)
async def deploy_monitoring(
    app_id: int,
    deploy_request: MonitoringDeployRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Deploy monitoring services for an app."""
    # Get parent app
    result = await db.execute(select(App).where(App.id == app_id))
    app = result.scalar_one_or_none()
    if not app:
        raise HTTPException(status_code=404, detail="App not found")

    # Check if app supports monitoring
    try:
        app_type = AppType(app.app_type)
        app_def = APP_DEFINITIONS.get(app_type, {})
        if not app_def.get("has_monitoring"):
            raise HTTPException(status_code=400, detail="This app does not support monitoring")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid app type")

    # Determine which services to deploy
    services_to_deploy = deploy_request.services or list(MONITORING_DEFINITIONS.keys())

    # Validate services
    for svc in services_to_deploy:
        if svc not in MONITORING_DEFINITIONS:
            raise HTTPException(status_code=400, detail=f"Unknown monitoring service: {svc}")

    # Check for already deployed services
    result = await db.execute(select(App).where(App.parent_app_id == app_id))
    existing = {a.app_type: a for a in result.scalars().all()}

    # Get worker
    await db.refresh(app, ["worker"])
    worker = app.worker

    # Find available ports starting from parent app's port + 10
    base_port = (app.port or 9000) + 10
    result = await db.execute(
        select(App.port).where(App.worker_id == worker.id, App.port.isnot(None))
    )
    used_ports = {row[0] for row in result.fetchall()}

    created_apps = []
    port = base_port
    for svc_type in services_to_deploy:
        if svc_type in existing:
            # Already deployed, skip
            created_apps.append(existing[svc_type])
            continue

        # Find next available port
        while port in used_ports:
            port += 1

        svc_def = MONITORING_DEFINITIONS[svc_type]
        svc_app = App(
            app_type=svc_type,
            name=f"{svc_def['name']} ({app.name})",
            worker_id=worker.id,
            parent_app_id=app_id,
            status=AppStatus.PENDING.value,
            proxy_path=f"/apps/{app.app_type}/monitoring/{svc_type}",
            port=port,
            use_proxy=app.use_proxy,
        )
        db.add(svc_app)
        await db.flush()
        created_apps.append(svc_app)
        used_ports.add(port)
        port += 1

    await db.commit()

    # Find prometheus port for Grafana configuration
    prometheus_port = None
    for svc_app in created_apps:
        if svc_app.app_type == "prometheus":
            prometheus_port = svc_app.port
            break

    # Start background deployment for new services
    # Deploy in order: prometheus first (Grafana needs it), then others
    deploy_order = ["prometheus", "grafana", "jaeger"]
    sorted_apps = sorted(
        created_apps,
        key=lambda a: deploy_order.index(a.app_type) if a.app_type in deploy_order else 99,
    )

    for svc_app in sorted_apps:
        if svc_app.status == AppStatus.PENDING.value:
            svc_def = MONITORING_DEFINITIONS[svc_app.app_type]
            background_tasks.add_task(
                deploy_monitoring_background,
                app_id=svc_app.id,
                parent_app_id=app_id,
                svc_type=svc_app.app_type,
                worker_address=worker.address,
                port=svc_app.port,
                svc_def=svc_def,
                use_proxy=svc_app.use_proxy,
                parent_app_port=app.port,
                prometheus_port=prometheus_port,
            )

    # Return status
    host = request.headers.get("host", "").split(":")[0]
    worker_host = worker.address.split(":")[0]
    if not host or host in ("localhost", "127.0.0.1"):
        host = worker_host

    services = []
    for svc_app in created_apps:
        await db.refresh(svc_app)
        url = (
            f"http://{host}:{svc_app.port}"
            if svc_app.port and svc_app.status == AppStatus.RUNNING.value
            else None
        )
        services.append(
            MonitoringServiceStatus(
                name=MONITORING_DEFINITIONS.get(svc_app.app_type, {}).get("name", svc_app.app_type),
                type=svc_app.app_type,
                status=svc_app.status,
                port=svc_app.port,
                url=url,
            )
        )

    return MonitoringStatus(enabled=True, services=services)


@router.delete("/{app_id}/monitoring")
async def remove_monitoring(
    app_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Remove all monitoring services for an app."""
    # Get parent app
    result = await db.execute(select(App).where(App.id == app_id))
    app = result.scalar_one_or_none()
    if not app:
        raise HTTPException(status_code=404, detail="App not found")

    # Get child monitoring apps
    result = await db.execute(select(App).where(App.parent_app_id == app_id))
    child_apps = result.scalars().all()

    if not child_apps:
        return {"message": "No monitoring services to remove"}

    # Get worker
    await db.refresh(app, ["worker"])
    worker = app.worker

    # Stop and remove containers
    async with httpx.AsyncClient(timeout=CONTAINER_ACTION_TIMEOUT) as client:
        for child in child_apps:
            if child.container_id:
                try:
                    # Stop container
                    await client.post(
                        f"http://{worker.address}/containers/{child.container_id}/stop"
                    )
                    # Remove container
                    await client.delete(f"http://{worker.address}/containers/{child.container_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove container for {child.app_type}: {e}")

            # Remove nginx proxy
            if child.use_proxy:
                try:
                    proxy_manager = get_proxy_manager()
                    await proxy_manager.remove_app_proxy(child.id)
                except Exception as e:
                    logger.warning(f"Failed to remove proxy for {child.app_type}: {e}")

            # Delete from database
            await db.delete(child)

    await db.commit()

    return {"message": f"Removed {len(child_apps)} monitoring service(s)"}


@router.delete("/{app_id}/monitoring/{service_type}")
async def remove_monitoring_service(
    app_id: int,
    service_type: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Remove a specific monitoring service."""
    # Get parent app
    result = await db.execute(select(App).where(App.id == app_id))
    app = result.scalar_one_or_none()
    if not app:
        raise HTTPException(status_code=404, detail="App not found")

    # Get specific monitoring app
    result = await db.execute(
        select(App).where(App.parent_app_id == app_id, App.app_type == service_type)
    )
    child = result.scalar_one_or_none()
    if not child:
        raise HTTPException(status_code=404, detail=f"Monitoring service {service_type} not found")

    # Get worker
    await db.refresh(app, ["worker"])
    worker = app.worker

    # Stop and remove container
    if child.container_id:
        async with httpx.AsyncClient(timeout=CONTAINER_ACTION_TIMEOUT) as client:
            try:
                await client.post(f"http://{worker.address}/containers/{child.container_id}/stop")
                await client.delete(f"http://{worker.address}/containers/{child.container_id}")
            except Exception as e:
                logger.warning(f"Failed to remove container for {service_type}: {e}")

    # Remove nginx proxy
    if child.use_proxy:
        try:
            proxy_manager = get_proxy_manager()
            await proxy_manager.remove_app_proxy(child.id)
        except Exception as e:
            logger.warning(f"Failed to remove proxy for {service_type}: {e}")

    # Delete from database
    await db.delete(child)
    await db.commit()

    return {"message": f"Removed {service_type} monitoring service"}


# =============================================================================
# Background Tasks
# =============================================================================


async def deploy_monitoring_background(
    app_id: int,
    parent_app_id: int,
    svc_type: str,
    worker_address: str,
    port: int,
    svc_def: dict,
    use_proxy: bool,
    parent_app_port: int | None = None,
    prometheus_port: int | None = None,
) -> None:
    """Background task to deploy a monitoring service.

    Args:
        app_id: Monitoring service app ID
        parent_app_id: Parent app (Semantic Router) ID
        svc_type: Service type (grafana, prometheus, jaeger)
        worker_address: Worker address
        port: Host port for this service
        svc_def: Service definition
        use_proxy: Whether to setup nginx proxy
        parent_app_port: Parent app's port (for calculating metrics port)
        prometheus_port: Prometheus port (for Grafana datasource config)
    """
    from app.database import async_session_maker

    async with async_session_maker() as db:
        try:
            # Get app
            result = await db.execute(select(App).where(App.id == app_id))
            app = result.scalar_one_or_none()
            if not app:
                logger.error(f"Monitoring app {app_id} not found")
                return

            # Get worker
            result = await db.execute(select(Worker).where(Worker.id == app.worker_id))
            worker = result.scalar_one_or_none()
            if not worker:
                logger.error(f"Worker not found for monitoring app {app_id}")
                app.status = AppStatus.ERROR.value
                app.status_message = "Worker not found"
                await db.commit()
                return

            # Get parent app for config
            parent_app = None
            if parent_app_id:
                result = await db.execute(select(App).where(App.id == parent_app_id))
                parent_app = result.scalar_one_or_none()

            # Phase 1: Pull image
            app.status = AppStatus.PULLING.value
            app.status_message = "Pulling image..."
            await db.commit()

            try:
                await pull_image_with_progress(worker, svc_def["image"], app_id)
            except Exception as e:
                app.status = AppStatus.ERROR.value
                app.status_message = f"Failed to pull image: {e}"
                await db.commit()
                return

            # Phase 2: Setup configuration (Prometheus needs config file)
            if svc_type == "prometheus" and parent_app:
                set_deployment_progress(app_id, "starting", 5, "Creating Prometheus config...")
                await _create_prometheus_config(worker_address, parent_app, port)

            # Phase 3: Create container
            app.status = AppStatus.STARTING.value
            app.status_message = "Starting container..."
            await db.commit()

            set_deployment_progress(app_id, "starting", 10, "Creating container...")

            container_id = await _create_monitoring_container(
                worker_address=worker_address,
                app_id=app_id,
                svc_type=svc_type,
                svc_def=svc_def,
                port=port,
                parent_app=parent_app,
                prometheus_port=prometheus_port,
            )
            if not container_id:
                return

            app.container_id = container_id
            await db.commit()

            # Phase 4: Wait for health
            set_deployment_progress(app_id, "starting", 50, "Waiting for service to start...")

            await wait_for_container_healthy(
                worker_address=worker_address,
                container_id=container_id,
                app_id=app_id,
                port=port,
            )

            # Phase 5: Post-deployment setup (Grafana needs datasource and dashboard)
            if svc_type == "grafana" and prometheus_port:
                set_deployment_progress(app_id, "starting", 80, "Configuring Grafana datasource...")
                await _configure_grafana(worker_address, port, prometheus_port)

            # Phase 6: Setup proxy
            if use_proxy:
                await _setup_monitoring_proxy(app_id, svc_type, worker_address, port)

            # Mark as running FIRST (before checking if all monitoring services are ready)
            app.status = AppStatus.RUNNING.value
            app.status_message = None
            await db.commit()

            set_deployment_progress(app_id, "running", 100, f"{svc_def['name']} deployed")
            logger.info(f"Monitoring service {svc_type} deployed for app {parent_app_id}")

            # Update parent app's environment with monitoring URLs
            # This must be called AFTER marking current service as running
            await _update_parent_monitoring_urls(db, parent_app_id)

        except Exception as e:
            logger.exception(f"Failed to deploy monitoring {svc_type}: {e}")
            try:
                result = await db.execute(select(App).where(App.id == app_id))
                app = result.scalar_one_or_none()
                if app:
                    app.status = AppStatus.ERROR.value
                    app.status_message = str(e)
                    await db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update monitoring error status: {db_error}")

            set_deployment_progress(app_id, "error", 0, str(e))


async def _create_monitoring_container(
    worker_address: str,
    app_id: int,
    svc_type: str,
    svc_def: dict,
    port: int,
    parent_app: App | None = None,
    prometheus_port: int | None = None,
) -> str | None:
    """Create monitoring container on worker.

    Args:
        worker_address: Worker address
        app_id: Monitoring app ID
        svc_type: Service type (grafana, prometheus, jaeger)
        svc_def: Service definition from MONITORING_DEFINITIONS
        port: Host port for this service
        parent_app: Parent app (Semantic Router) for getting metrics endpoint
        prometheus_port: Prometheus port (needed for Grafana datasource config)
    """
    from app.database import async_session_maker

    container_name = f"lmstack-monitoring-{svc_type}"
    worker_host = worker_address.split(":")[0]

    # Build volumes
    volumes = []
    for vol in svc_def.get("volumes", []):
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
            "container_port": svc_def["internal_port"],
            "host_port": port,
            "protocol": "tcp",
        }
    ]

    # Build env vars
    env_vars = dict(svc_def.get("env_template", {}))

    # Service-specific configuration
    command = None

    if svc_type == "prometheus" and parent_app:
        # Prometheus needs to know where to scrape metrics from
        # Semantic Router exposes metrics on port 9190
        parent_metrics_port = (
            parent_app.port + 2 if parent_app.port else 9192
        )  # Main port + 2 = metrics

        # For local workers, use host.docker.internal
        # For remote workers, use the worker's actual address
        if worker_host in ("localhost", "127.0.0.1"):
            metrics_target = f"host.docker.internal:{parent_metrics_port}"
        else:
            metrics_target = f"{worker_host}:{parent_metrics_port}"

        # Prometheus config passed via command line args
        # We use a minimal config that scrapes the semantic router
        command = [
            "--config.file=/etc/prometheus/prometheus.yml",
            "--storage.tsdb.path=/prometheus",
            "--web.console.libraries=/usr/share/prometheus/console_libraries",
            "--web.console.templates=/usr/share/prometheus/consoles",
            "--web.enable-lifecycle",
        ]

        # We'll need to create the config file via init container or bind mount
        # For now, use file_configs volume approach - create config on worker
        env_vars["ROUTER_TARGET"] = metrics_target

    elif svc_type == "grafana" and prometheus_port:
        # Grafana needs to know where Prometheus is
        if worker_host in ("localhost", "127.0.0.1"):
            prometheus_url = f"http://host.docker.internal:{prometheus_port}"
        else:
            prometheus_url = f"http://{worker_host}:{prometheus_port}"

        # Use Grafana's environment-based datasource provisioning
        env_vars["GF_DATASOURCES_DEFAULT_NAME"] = "Prometheus"
        env_vars["GF_DATASOURCES_DEFAULT_TYPE"] = "prometheus"
        env_vars["GF_DATASOURCES_DEFAULT_URL"] = prometheus_url
        env_vars["GF_DATASOURCES_DEFAULT_ACCESS"] = "proxy"
        env_vars["GF_DATASOURCES_DEFAULT_ISDEFAULT"] = "true"

        # CRITICAL: Set Grafana's root URL so redirects work correctly
        # Without this, Grafana redirects to localhost:3000 which breaks iframe embedding
        env_vars["GF_SERVER_ROOT_URL"] = f"http://{worker_host}:{port}"
        # Also set serve_from_sub_path since we access via proxy
        env_vars["GF_SERVER_SERVE_FROM_SUB_PATH"] = "false"

    payload = {
        "name": container_name,
        "image": svc_def["image"],
        "env": env_vars,
        "ports": ports,
        "volumes": volumes,
        "restart_policy": "unless-stopped",
        "labels": {
            "lmstack.monitoring": "true",
            "lmstack.monitoring.type": svc_type,
            "lmstack.monitoring.app_id": str(app_id),
        },
        "extra_hosts": {"host.docker.internal": "host-gateway"},
    }

    if command:
        payload["command"] = command

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


async def _setup_monitoring_proxy(
    app_id: int,
    svc_type: str,
    worker_address: str,
    port: int,
) -> None:
    """Setup nginx proxy for monitoring service."""
    set_deployment_progress(app_id, "starting", 95, "Setting up proxy...")

    try:
        proxy_manager = get_proxy_manager()
        proxy_worker_host = worker_address.split(":")[0]

        await proxy_manager.add_app_proxy(
            app_id=app_id,
            app_type=f"monitoring-{svc_type}",
            listen_port=port,
            worker_host=proxy_worker_host,
            worker_port=port,
        )
        logger.info(f"Nginx proxy configured for monitoring {svc_type} on port {port}")

    except Exception as e:
        logger.warning(f"Failed to setup nginx proxy for monitoring: {e}")


async def _update_parent_monitoring_urls(db: AsyncSession, parent_app_id: int) -> None:
    """Update parent app's environment with monitoring URLs.

    This updates the Semantic Router container with the URLs of deployed monitoring services.
    When all monitoring services are running, it restarts the parent container with new env vars.
    """
    # IMPORTANT: Use a fresh session to see commits from other parallel background tasks
    # Each background task has its own isolated session, so we need a new one to get
    # the current state of all monitoring services
    from app.database import async_session_maker

    async with async_session_maker() as fresh_db:
        # Get parent app
        result = await fresh_db.execute(select(App).where(App.id == parent_app_id))
        parent_app = result.scalar_one_or_none()
        if not parent_app:
            return

        # Get all monitoring services
        result = await fresh_db.execute(select(App).where(App.parent_app_id == parent_app_id))
        monitoring_apps = list(result.scalars().all())

        # Build monitoring URLs
        # These URLs are used by the Semantic Router DASHBOARD (runs in user's browser)
        # to render iframes for Grafana/Jaeger. Since the browser can't resolve
        # host.docker.internal, we must use the worker's external IP.
        await fresh_db.refresh(parent_app, ["worker"])

        # Use the worker's external IP for browser-accessible URLs
        worker_host = parent_app.worker.address.split(":")[0]

        monitoring_urls = {}
        all_running = True
        for mon_app in monitoring_apps:
            if mon_app.status == AppStatus.RUNNING.value and mon_app.port:
                if mon_app.app_type == "grafana":
                    monitoring_urls["grafana_url"] = f"http://{worker_host}:{mon_app.port}"
                elif mon_app.app_type == "prometheus":
                    monitoring_urls["prometheus_url"] = f"http://{worker_host}:{mon_app.port}"
                elif mon_app.app_type == "jaeger":
                    monitoring_urls["jaeger_url"] = f"http://{worker_host}:{mon_app.port}"
            elif mon_app.status not in (AppStatus.ERROR.value, AppStatus.STOPPED.value):
                all_running = False

        # Store in parent app's config for reference
        config = parent_app.config or {}
        config["monitoring_urls"] = monitoring_urls
        parent_app.config = config
        await fresh_db.commit()

        logger.info(f"Updated monitoring URLs for app {parent_app_id}: {monitoring_urls}")
        logger.info(
            f"Restart check: all_running={all_running}, has_urls={bool(monitoring_urls)}, "
            f"parent_status={parent_app.status}, services={[(m.app_type, m.status) for m in monitoring_apps]}"
        )

        # If all monitoring services are running, restart parent app to pick up new URLs
        if all_running and monitoring_urls and parent_app.status == AppStatus.RUNNING.value:
            logger.info(
                f"All monitoring services running, restarting parent app {parent_app_id} to apply URLs"
            )
            try:
                await _restart_parent_with_monitoring_urls(fresh_db, parent_app, monitoring_urls)
            except Exception as e:
                logger.exception(f"Failed to restart parent app with monitoring URLs: {e}")


async def _restart_parent_with_monitoring_urls(
    db: AsyncSession,
    parent_app: App,
    monitoring_urls: dict,
) -> None:
    """Restart parent app container with monitoring URLs injected into environment.

    This stops the old container, creates a new one with updated env vars, and starts it.
    """
    from app.api.apps.deployment import wait_for_container_healthy
    from app.models.app import APP_DEFINITIONS, AppType

    if not parent_app.container_id:
        return

    try:
        app_type = AppType(parent_app.app_type)
        app_def = APP_DEFINITIONS.get(app_type)
        if not app_def:
            return
    except ValueError:
        return

    await db.refresh(parent_app, ["worker"])
    worker = parent_app.worker
    worker_address = worker.address

    logger.info(f"Restarting {parent_app.name} with monitoring URLs: {monitoring_urls}")

    async with httpx.AsyncClient(timeout=CONTAINER_ACTION_TIMEOUT) as client:
        # Stop and remove old container
        try:
            await client.post(f"http://{worker_address}/containers/{parent_app.container_id}/stop")
            await client.delete(f"http://{worker_address}/containers/{parent_app.container_id}")
        except Exception as e:
            logger.warning(f"Failed to stop/remove old container: {e}")

        # Build env vars from template, injecting monitoring URLs
        new_env = {}
        for key, value in app_def.get("env_template", {}).items():
            if value == "{grafana_url}":
                new_env[key] = monitoring_urls.get("grafana_url", "")
            elif value == "{prometheus_url}":
                new_env[key] = monitoring_urls.get("prometheus_url", "")
            elif value == "{jaeger_url}":
                new_env[key] = monitoring_urls.get("jaeger_url", "")
            elif value == "{hf_token}":
                # Try to get HF_TOKEN from app config if stored, otherwise empty
                app_config = parent_app.config or {}
                new_env[key] = app_config.get("hf_token", "")
            elif value.startswith("{") and value.endswith("}"):
                # Other placeholders - use empty or defaults
                new_env[key] = ""
            else:
                # Static values
                new_env[key] = value

        # Rebuild container with same config but new env
        container_name = f"lmstack-app-{app_type.value}"

        volumes = []
        for vol in app_def.get("volumes", []):
            volumes.append(
                {
                    "source": f"{container_name}-{vol['name']}",
                    "destination": vol["destination"],
                    "mode": "rw",
                }
            )

        ports = [
            {
                "container_port": app_def["internal_port"],
                "host_port": parent_app.port,
                "protocol": "tcp",
            }
        ]

        # Add additional ports
        for i, port_info in enumerate(app_def.get("additional_ports", [])):
            if isinstance(port_info, dict):
                container_port = port_info["container_port"]
            else:
                container_port = port_info
            ports.append(
                {
                    "container_port": container_port,
                    "host_port": parent_app.port + 1 + i,
                    "protocol": "tcp",
                }
            )

        payload = {
            "name": container_name,
            "image": app_def["image"],
            "env": new_env,
            "ports": ports,
            "volumes": volumes,
            "restart_policy": "unless-stopped",
            "labels": {
                "lmstack.app": "true",
                "lmstack.app.type": app_type.value,
                "lmstack.app.id": str(parent_app.id),
            },
            "extra_hosts": {"host.docker.internal": "host-gateway"},
        }

        if app_def.get("entrypoint"):
            payload["entrypoint"] = app_def["entrypoint"]
        if app_def.get("command"):
            payload["command"] = app_def["command"]
        if app_def.get("cap_add"):
            payload["cap_add"] = app_def["cap_add"]

        try:
            resp = await client.post(f"http://{worker_address}/containers", json=payload)
            if resp.status_code >= 400:
                logger.error(f"Failed to create new container: {resp.text}")
                return

            new_container_id = resp.json().get("id")
            parent_app.container_id = new_container_id
            await db.commit()

            logger.info(f"Created new container {new_container_id} with monitoring URLs")

            # Wait for container to be healthy
            await wait_for_container_healthy(
                worker_address=worker_address,
                container_id=new_container_id,
                app_id=parent_app.id,
                port=parent_app.port,
            )

            logger.info(f"Parent app {parent_app.id} restarted successfully with monitoring URLs")

        except Exception as e:
            logger.error(f"Failed to restart parent app: {e}")


async def _create_prometheus_config(
    worker_address: str,
    parent_app: App,
    prometheus_port: int,
) -> None:
    """Create Prometheus configuration file on worker.

    Uses a helper container to write the prometheus.yml config to the volume.
    """
    # Calculate metrics port: Semantic Router main port + 2 = metrics port (9190 internally mapped)
    # The router exposes :8888 (API), :8700 (Dashboard), and :9190 (metrics)
    # We map them as: port, port+1, port+2
    metrics_port = parent_app.port + 2 if parent_app.port else 9192

    # Prometheus runs in a container, so it must use host.docker.internal
    # to access services on the same host (regardless of worker's external IP)
    metrics_target = f"host.docker.internal:{metrics_port}"

    # Prometheus configuration YAML - using cat with heredoc to avoid quote issues
    prometheus_config = f"""global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: semantic-router
    static_configs:
      - targets: ["{metrics_target}"]
    metrics_path: /metrics
    scrape_interval: 5s
"""

    volume_name = "lmstack-monitoring-prometheus-prometheus-config"

    # Use a helper container to write the config file
    # Using base64 encoding to avoid shell quoting issues
    config_b64 = base64.b64encode(prometheus_config.encode()).decode()

    helper_payload = {
        "name": "lmstack-prometheus-config-helper",
        "image": "alpine:latest",
        "command": [
            "sh",
            "-c",
            f"mkdir -p /etc/prometheus && echo '{config_b64}' | base64 -d > /etc/prometheus/prometheus.yml && cat /etc/prometheus/prometheus.yml",
        ],
        "volumes": [
            {
                "source": volume_name,
                "destination": "/etc/prometheus",
                "mode": "rw",
            }
        ],
        "restart_policy": "no",
        "labels": {"lmstack.helper": "true"},
    }

    async with httpx.AsyncClient(timeout=CONTAINER_ACTION_TIMEOUT) as client:
        try:
            # Create and run helper container
            resp = await client.post(
                f"http://{worker_address}/containers",
                json=helper_payload,
            )
            if resp.status_code >= 400:
                logger.warning(f"Failed to create prometheus config helper: {resp.text}")
                return

            helper_id = resp.json().get("id")
            logger.info(f"Created prometheus config with helper container {helper_id}")

            # Wait a moment for the container to finish
            await asyncio.sleep(2)

            # Clean up helper container
            try:
                await client.delete(
                    f"http://{worker_address}/containers/{helper_id}",
                    params={"force": True},
                )
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Failed to create prometheus config: {e}")


async def _configure_grafana(
    worker_address: str,
    grafana_port: int,
    prometheus_port: int,
) -> None:
    """Configure Grafana datasource and dashboard via API.

    Grafana needs a moment to start, so we retry a few times.
    """
    import json
    from pathlib import Path

    worker_host = worker_address.split(":")[0]

    # Grafana runs in a container, so it must use host.docker.internal
    # to access Prometheus on the same host
    prometheus_url = f"http://host.docker.internal:{prometheus_port}"

    # For API calls from the backend, use the actual host
    if worker_host in ("localhost", "127.0.0.1"):
        grafana_url = f"http://localhost:{grafana_port}"
    else:
        grafana_url = f"http://{worker_host}:{grafana_port}"

    # Grafana datasource payload
    datasource_payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": prometheus_url,
        "access": "proxy",
        "isDefault": True,
    }

    datasource_uid = None

    # Try to add datasource (Grafana needs time to start)
    max_retries = 5
    for i in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Check if Grafana is ready
                health_resp = await client.get(f"{grafana_url}/api/health")
                if health_resp.status_code != 200:
                    raise Exception("Grafana not ready")

                # Add datasource
                ds_resp = await client.post(
                    f"{grafana_url}/api/datasources",
                    json=datasource_payload,
                    auth=("admin", "admin"),
                )

                if ds_resp.status_code == 200:
                    ds_data = ds_resp.json()
                    datasource_uid = ds_data.get("datasource", {}).get("uid")
                    logger.info(f"Grafana datasource created: uid={datasource_uid}")
                    break
                elif ds_resp.status_code == 409:  # Already exists
                    # Get existing datasource UID
                    get_ds_resp = await client.get(
                        f"{grafana_url}/api/datasources/name/Prometheus",
                        auth=("admin", "admin"),
                    )
                    if get_ds_resp.status_code == 200:
                        datasource_uid = get_ds_resp.json().get("uid")
                        logger.info(f"Grafana datasource exists: uid={datasource_uid}")
                    break
                else:
                    logger.warning(f"Failed to add Grafana datasource: {ds_resp.text}")

        except Exception as e:
            logger.debug(f"Grafana not ready yet (attempt {i+1}/{max_retries}): {e}")
            await asyncio.sleep(3)

    if not datasource_uid:
        logger.warning("Failed to configure Grafana datasource after retries")
        return

    # Import LLM Router dashboard
    dashboard_path = Path(
        "/home/rickychen/Desktop/llm/lmstack/semantic-router/deploy/docker-compose/addons/llm-router-dashboard.json"
    )
    if not dashboard_path.exists():
        logger.warning(f"Dashboard file not found: {dashboard_path}")
        return

    try:
        dashboard_json = json.loads(dashboard_path.read_text())

        # Replace datasource variable with actual UID
        dashboard_str = json.dumps(dashboard_json)
        dashboard_str = dashboard_str.replace("${DS_PROMETHEUS}", datasource_uid)
        dashboard_str = dashboard_str.replace('"uid": "prometheus"', f'"uid": "{datasource_uid}"')
        dashboard_json = json.loads(dashboard_str)

        # Remove id to create new dashboard
        dashboard_json.pop("id", None)
        dashboard_json["uid"] = "llm-router-metrics"

        # Import dashboard
        import_payload = {
            "dashboard": dashboard_json,
            "overwrite": True,
            "inputs": [
                {
                    "name": "DS_PROMETHEUS",
                    "type": "datasource",
                    "pluginId": "prometheus",
                    "value": datasource_uid,
                }
            ],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            dash_resp = await client.post(
                f"{grafana_url}/api/dashboards/import",
                json=import_payload,
                auth=("admin", "admin"),
            )

            if dash_resp.status_code == 200:
                logger.info("LLM Router dashboard imported successfully")
            else:
                logger.warning(f"Failed to import dashboard: {dash_resp.text}")

    except Exception as e:
        logger.exception(f"Failed to import Grafana dashboard: {e}")

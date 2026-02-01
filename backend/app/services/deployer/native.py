"""Native Mac deployment operations.

This module handles native deployment operations for macOS,
including Ollama, MLX, and llama.cpp backends.
"""

import asyncio
import logging

import httpx

from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import BackendType

logger = logging.getLogger(__name__)


async def deploy_native(deployment: Deployment, db) -> dict:
    """Deploy using native backend (Mac without Docker).

    Supports Ollama, MLX, and llama.cpp backends on macOS.
    """
    # Import here to avoid circular imports
    from app.services.deployer.health import wait_for_native_api_ready

    worker = deployment.worker
    model = deployment.model
    backend = deployment.backend

    # Validate backend is supported
    available_backends = worker.available_backends
    if backend not in available_backends:
        return {
            "error": f"Backend '{backend}' not available on this worker. "
            f"Available backends: {', '.join(available_backends)}"
        }

    try:
        worker_url = f"http://{worker.effective_address}/native/deploy"

        deploy_request = {
            "deployment_id": deployment.id,
            "deployment_name": deployment.name,
            "model_id": model.model_id,
            "backend": backend,
            "port": 0,  # Auto-assign
            "extra_params": deployment.extra_params,
        }

        deployment.status_message = f"Starting {backend} deployment..."
        await db.commit()

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(worker_url, json=deploy_request)

            if response.status_code != 200:
                error_detail = response.json().get("detail", response.text)
                return {"error": f"Native deployment failed: {error_detail}"}

            result = response.json()
            deployment.port = result.get("port")
            # Use process_id as container_id for native deployments
            deployment.container_id = result.get("process_id")

        # Wait for API to be ready
        deployment.status_message = "Waiting for model to be ready..."
        await db.commit()

        # For Ollama on native, the API is at port 11434
        if backend == BackendType.OLLAMA.value:
            api_port = 11434
        else:
            api_port = deployment.port

        api_ready = await wait_for_native_api_ready(
            worker.effective_address,
            api_port,
            deployment.id,
            db,
            backend=backend,
        )

        if api_ready is None:
            return {}  # Cancelled
        elif api_ready:
            deployment.status = DeploymentStatus.RUNNING.value
            deployment.status_message = "Model ready"
            asyncio.create_task(_update_semantic_router_config_background())
        else:
            deployment.status = DeploymentStatus.ERROR.value
            deployment.status_message = "API failed to start"

        await db.commit()
        return {}

    except httpx.ConnectError as e:
        return {"error": f"Cannot connect to worker: {e}"}
    except Exception as e:
        logger.exception(f"Native deployment error: {e}")
        return {"error": str(e)}


async def _update_semantic_router_config_background():
    """Background task to update semantic router config after deployment changes."""
    try:
        from app.api.apps.deployment import update_semantic_router_config_if_deployed
        from app.database import async_session_maker

        async with async_session_maker() as db:
            await update_semantic_router_config_if_deployed(db)
    except Exception as e:
        logger.debug(f"Failed to update semantic router config: {e}")

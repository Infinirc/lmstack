"""DeployerService - Main orchestration for model deployment.

This module contains the main DeployerService class that orchestrates
model deployment operations across different backends and workers.
"""

import asyncio
import logging

import httpx
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import async_session_maker
from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import BackendType
from app.models.worker import OSType

from .config import build_deploy_request
from .docker import deploy_local, get_logs_local, image_exists_local, pull_image_local, stop_local
from .health import (
    HEALTH_CHECK_INTERVAL,
    HEALTH_CHECK_REQUEST_TIMEOUT,
    HEALTH_CHECK_SLOW_THRESHOLD,
    ollama_pull_model,
    wait_for_api_ready,
    wait_for_ollama_ready,
)
from .native import deploy_native

logger = logging.getLogger(__name__)


async def _update_semantic_router_config_background():
    """Background task to update semantic router config after deployment changes."""
    try:
        from app.api.apps.deployment import update_semantic_router_config_if_deployed

        async with async_session_maker() as db:
            await update_semantic_router_config_if_deployed(db)
    except Exception as e:
        logger.debug(f"Failed to update semantic router config: {e}")


class DeployerService:
    """Service for deploying models to workers"""

    # Health check configuration (exposed as class attributes for backwards compatibility)
    HEALTH_CHECK_INTERVAL = HEALTH_CHECK_INTERVAL
    HEALTH_CHECK_SLOW_THRESHOLD = HEALTH_CHECK_SLOW_THRESHOLD
    HEALTH_CHECK_REQUEST_TIMEOUT = HEALTH_CHECK_REQUEST_TIMEOUT

    async def deploy(self, deployment_id: int) -> None:
        """Deploy a model to a worker"""
        async with async_session_maker() as db:
            result = await db.execute(
                select(Deployment)
                .where(Deployment.id == deployment_id)
                .options(
                    selectinload(Deployment.worker),
                    selectinload(Deployment.model),
                )
            )
            deployment = result.scalar_one_or_none()

            if not deployment:
                logger.error(f"Deployment {deployment_id} not found")
                return

            try:
                # Update status to starting
                deployment.status = DeploymentStatus.STARTING.value
                deployment.status_message = "Sending deployment request to worker..."
                await db.commit()

                # Check if worker supports Docker or needs native deployment
                worker = deployment.worker
                backend = deployment.backend

                # Mac with Ollama, MLX, llama_cpp, or vLLM should use native deployment
                # vLLM on Mac uses vLLM-Metal (native Apple Silicon acceleration)
                # Mac without Docker should also use native deployment
                is_mac = worker.os_type == OSType.DARWIN.value
                native_backends = (
                    BackendType.OLLAMA.value,
                    BackendType.MLX.value,
                    BackendType.LLAMA_CPP.value,
                    BackendType.VLLM.value,  # vLLM-Metal on Mac
                )
                is_mac_native = is_mac and (
                    backend in native_backends or not worker.supports_docker
                )

                # Use native deployment for Mac
                if is_mac_native:
                    result = await deploy_native(deployment, db)
                    if result.get("error"):
                        deployment.status = DeploymentStatus.ERROR.value
                        deployment.status_message = result["error"]
                        await db.commit()
                    return

                # Build deployment request
                deploy_request = build_deploy_request(deployment)

                # Check if this is a local worker
                is_local = self._is_local_worker(deployment.worker.address)

                if is_local:
                    # Check if image needs to be pulled
                    image = deploy_request["image"]
                    if not image_exists_local(image):
                        deployment.status_message = f"Pulling image: {image}..."
                        await db.commit()

                        pull_success = await pull_image_local(image)
                        if not pull_success:
                            deployment.status = DeploymentStatus.ERROR.value
                            deployment.status_message = f"Failed to pull image: {image}"
                            await db.commit()
                            return

                    deployment.status_message = "Starting container..."
                    await db.commit()

                    # Deploy locally using Docker directly
                    result = await deploy_local(deploy_request)
                    if result.get("error"):
                        deployment.status = DeploymentStatus.ERROR.value
                        deployment.status_message = result["error"]
                        await db.commit()
                        return
                    deployment.container_id = result.get("container_id")
                    deployment.port = result.get("port")
                    # Store container_name for internal Docker network communication
                    local_container_name = result.get("container_name")
                    deployment.container_name = local_container_name
                else:
                    local_container_name = None  # Remote workers use IP:port
                    # Send to remote worker agent
                    worker_url = f"http://{deployment.worker.address}/deploy"
                    progress_url = (
                        f"http://{deployment.worker.address}/pull-progress/{deployment.id}"
                    )

                    # Start deployment request and poll for progress
                    async with httpx.AsyncClient(timeout=300.0) as client:
                        # Start the deployment in a task
                        deploy_task = asyncio.create_task(
                            client.post(worker_url, json=deploy_request)
                        )

                        # Poll for progress while waiting
                        while not deploy_task.done():
                            try:
                                progress_resp = await client.get(progress_url, timeout=5.0)
                                if progress_resp.status_code == 200:
                                    progress_data = progress_resp.json()
                                    status = progress_data.get("status", "")
                                    image = progress_data.get("image", "")
                                    progress = progress_data.get("progress", 0)

                                    if status == "pulling":
                                        deployment.status_message = (
                                            f"Pulling image {image}... ({progress}%)"
                                        )
                                        await db.commit()
                                    elif status == "completed":
                                        deployment.status_message = (
                                            "Image pulled, starting container..."
                                        )
                                        await db.commit()
                                    elif status == "starting":
                                        deployment.status_message = "Starting container..."
                                        await db.commit()
                            except Exception:
                                pass  # Progress polling is best-effort

                            await asyncio.sleep(2)

                        response = await deploy_task

                        if response.status_code != 200:
                            deployment.status = DeploymentStatus.ERROR.value
                            deployment.status_message = f"Worker returned error: {response.text}"
                            await db.commit()
                            return

                        result_data = response.json()
                        deployment.container_id = result_data.get("container_id")
                        deployment.port = result_data.get("port")

                # Container started, now waiting for model to load
                deployment.status = DeploymentStatus.STARTING.value
                deployment.status_message = "Downloading model and Loading model into GPU memory..."
                await db.commit()

                # For Ollama, we need to pull the model first
                if deployment.backend == BackendType.OLLAMA.value:
                    deployment.status_message = "Waiting for Ollama container to start..."
                    await db.commit()

                    # Wait for Ollama API to be available before pulling
                    ollama_ready = await wait_for_ollama_ready(
                        deployment.worker.address,
                        deployment.port,
                        container_name=local_container_name,
                    )
                    if not ollama_ready:
                        deployment.status = DeploymentStatus.ERROR.value
                        deployment.status_message = "Ollama container failed to start"
                        await db.commit()
                        return

                    deployment.status_message = "Pulling model with Ollama..."
                    await db.commit()

                    pull_success = await ollama_pull_model(
                        deployment.worker.address,
                        deployment.port,
                        deployment.model.model_id,
                        container_name=local_container_name,
                    )
                    if not pull_success:
                        deployment.status = DeploymentStatus.ERROR.value
                        deployment.status_message = "Failed to pull model with Ollama"
                        await db.commit()
                        return

                    deployment.status_message = "Model pulled, waiting for API..."
                    await db.commit()

                # Wait for the API endpoint to become ready
                api_ready = await wait_for_api_ready(
                    deployment.worker.address,
                    deployment.port,
                    deployment_id,
                    db,
                    backend=deployment.backend,
                    container_name=local_container_name,
                )

                # Refresh deployment object after health check updates
                await db.refresh(deployment)

                if api_ready is None:
                    # Deployment was cancelled, don't update status
                    logger.info(f"Deployment {deployment_id} cancelled during startup")
                    return
                else:
                    # api_ready is True, model is ready
                    deployment.status = DeploymentStatus.RUNNING.value
                    deployment.status_message = "Model ready"

                    # Update semantic router config if deployed
                    asyncio.create_task(_update_semantic_router_config_background())

            except httpx.ConnectError:
                deployment.status = DeploymentStatus.ERROR.value
                deployment.status_message = (
                    f"Cannot connect to worker at {deployment.worker.address}"
                )
            except Exception as e:
                logger.exception(f"Error deploying {deployment_id}")
                deployment.status = DeploymentStatus.ERROR.value
                deployment.status_message = str(e)

            await db.commit()

    def _is_local_worker(self, address: str) -> bool:
        """Check if the worker address refers to the local machine."""
        if not address:
            return False
        host = address.split(":")[0].lower()
        return host in ("localhost", "127.0.0.1", "local")

    async def stop(self, deployment_id: int) -> None:
        """Stop a deployment"""
        async with async_session_maker() as db:
            result = await db.execute(
                select(Deployment)
                .where(Deployment.id == deployment_id)
                .options(selectinload(Deployment.worker))
            )
            deployment = result.scalar_one_or_none()

            if not deployment or not deployment.container_id:
                return

            try:
                worker = deployment.worker

                # Check if this is a native deployment (process_id starts with "native-")
                is_native = deployment.container_id.startswith("native-")

                if is_native:
                    # Stop native process
                    worker_url = f"http://{worker.effective_address}/native/stop"
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        await client.post(
                            worker_url,
                            json={"process_id": deployment.container_id},
                        )
                else:
                    # Docker-based deployment
                    is_local = self._is_local_worker(worker.address)

                    if is_local:
                        # Stop locally using Docker directly
                        await stop_local(deployment.container_id)
                    else:
                        worker_url = f"http://{worker.address}/stop"

                        async with httpx.AsyncClient(timeout=60.0) as client:
                            await client.post(
                                worker_url, json={"container_id": deployment.container_id}
                            )

            except Exception as e:
                logger.warning(f"Error stopping deployment {deployment_id}: {e}")

    async def get_logs(self, deployment: Deployment, tail: int = 100) -> str:
        """Get logs from a deployment"""
        if not deployment.container_id or not deployment.worker:
            return "No deployment process running"

        try:
            worker = deployment.worker

            # Check if this is a native deployment
            is_native = deployment.container_id.startswith("native-")

            if is_native:
                # Get logs from native process
                worker_url = (
                    f"http://{worker.effective_address}/native/logs/{deployment.container_id}"
                )
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(worker_url, params={"tail": tail})
                    if response.status_code == 200:
                        return response.json().get("logs", "")
                    else:
                        return f"Error fetching logs: {response.text}"
            else:
                # Docker-based deployment
                is_local = self._is_local_worker(worker.address)

                if is_local:
                    return get_logs_local(deployment.container_id, tail)
                else:
                    worker_url = f"http://{worker.address}/logs"

                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(
                            worker_url,
                            params={
                                "container_id": deployment.container_id,
                                "tail": tail,
                            },
                        )

                        if response.status_code == 200:
                            return response.json().get("logs", "")
                        else:
                            return f"Error fetching logs: {response.text}"

        except httpx.ConnectError:
            return f"Cannot connect to worker at {deployment.worker.address}"
        except Exception as e:
            return f"Error: {str(e)}"

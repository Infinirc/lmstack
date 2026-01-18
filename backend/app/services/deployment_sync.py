"""Deployment Sync Service

Synchronizes deployment status with actual container state.
This is important after system reboot to ensure database status matches reality.
"""

import asyncio
import logging

import docker
import httpx
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import async_session_maker
from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import BackendType

logger = logging.getLogger(__name__)


class DeploymentSyncService:
    """Service for synchronizing deployment status with actual container state."""

    # Health check configuration
    HEALTH_CHECK_TIMEOUT = 10  # seconds
    CONTAINER_CHECK_TIMEOUT = 5  # seconds
    MAX_CONCURRENT_CHECKS = 5  # limit concurrent health checks

    def __init__(self):
        self._docker_client: docker.DockerClient | None = None

    @property
    def docker_client(self) -> docker.DockerClient:
        """Lazy-load Docker client."""
        if self._docker_client is None:
            try:
                self._docker_client = docker.from_env()
            except docker.errors.DockerException as e:
                logger.warning(f"Failed to connect to Docker: {e}")
                raise
        return self._docker_client

    async def sync_all_deployments(self) -> dict:
        """Synchronize all deployment statuses on startup.

        Returns:
            dict with sync statistics
        """
        logger.info("Starting deployment status synchronization...")

        stats = {
            "total": 0,
            "running_verified": 0,
            "restarting": 0,
            "container_missing": 0,
            "api_not_ready": 0,
            "errors": 0,
            "skipped": 0,
        }

        async with async_session_maker() as db:
            # Get all deployments that should be running
            result = await db.execute(
                select(Deployment)
                .where(
                    Deployment.status.in_(
                        [
                            DeploymentStatus.RUNNING.value,
                            DeploymentStatus.STARTING.value,
                        ]
                    )
                )
                .options(
                    selectinload(Deployment.worker),
                    selectinload(Deployment.model),
                )
            )
            deployments = result.scalars().all()
            stats["total"] = len(deployments)

            if not deployments:
                logger.info("No active deployments to sync")
                return stats

            logger.info(f"Found {len(deployments)} active deployments to check")

            # Check deployments with limited concurrency
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CHECKS)

            async def check_with_semaphore(deployment: Deployment):
                async with semaphore:
                    return await self._check_and_update_deployment(deployment, db)

            tasks = [check_with_semaphore(d) for d in deployments]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Deployment check failed: {result}")
                    stats["errors"] += 1
                elif isinstance(result, str):
                    if result == "running_verified":
                        stats["running_verified"] += 1
                    elif result == "restarting":
                        stats["restarting"] += 1
                    elif result == "container_missing":
                        stats["container_missing"] += 1
                    elif result == "api_not_ready":
                        stats["api_not_ready"] += 1
                    elif result == "skipped":
                        stats["skipped"] += 1

            await db.commit()

        logger.info(
            f"Deployment sync complete: {stats['running_verified']} running, "
            f"{stats['restarting']} restarting, {stats['container_missing']} missing, "
            f"{stats['api_not_ready']} not ready, {stats['errors']} errors"
        )

        return stats

    async def _check_and_update_deployment(self, deployment: Deployment, db) -> str:
        """Check a single deployment and update its status.

        Returns:
            Status string: running_verified, restarting, container_missing, api_not_ready, skipped
        """
        logger.debug(f"Checking deployment {deployment.id}: {deployment.name}")

        if not deployment.worker:
            logger.warning(f"Deployment {deployment.id} has no worker, skipping")
            return "skipped"

        if not deployment.container_id:
            # If deployment is still starting, skip it
            if deployment.status == DeploymentStatus.STARTING.value:
                logger.debug(f"Deployment {deployment.id} is still starting, skipping")
                return "skipped"
            # Only mark as error if deployment claims to be RUNNING but has no container
            logger.warning(f"Deployment {deployment.id} has no container_id, marking as error")
            deployment.status = DeploymentStatus.ERROR.value
            deployment.status_message = "Container ID missing after restart"
            return "container_missing"

        # Check if this is a local worker
        is_local = self._is_local_worker(deployment.worker.address)

        if is_local:
            return await self._check_local_deployment(deployment)
        else:
            return await self._check_remote_deployment(deployment)

    def _is_local_worker(self, address: str) -> bool:
        """Check if the worker address refers to the local machine."""
        if not address:
            return False
        host = address.split(":")[0].lower()
        return host in ("localhost", "127.0.0.1", "local")

    async def _check_local_deployment(self, deployment: Deployment) -> str:
        """Check a local deployment's container and API status."""
        try:
            # Check container status
            container = self.docker_client.containers.get(deployment.container_id)
            container_status = container.status

            if container_status == "running":
                # Container is running, check API health
                api_healthy = await self._check_api_health(
                    deployment.worker.address,
                    deployment.port,
                    deployment.backend,
                    deployment.container_name,
                )

                if api_healthy:
                    # Everything is good
                    if deployment.status != DeploymentStatus.RUNNING.value:
                        deployment.status = DeploymentStatus.RUNNING.value
                        deployment.status_message = "Model ready (verified after restart)"
                    logger.info(f"Deployment {deployment.name}: running and healthy")
                    return "running_verified"
                else:
                    # Container running but API not ready yet
                    deployment.status = DeploymentStatus.STARTING.value
                    deployment.status_message = "Container running, waiting for model to load..."
                    logger.info(f"Deployment {deployment.name}: container running, API not ready")
                    return "api_not_ready"

            elif container_status == "restarting":
                deployment.status = DeploymentStatus.STARTING.value
                deployment.status_message = "Container restarting after system reboot..."
                logger.info(f"Deployment {deployment.name}: container restarting")
                return "restarting"

            elif container_status in ("exited", "dead"):
                # Container exists but stopped - try to restart it
                logger.info(
                    f"Deployment {deployment.name}: container {container_status}, attempting restart"
                )
                try:
                    container.start()
                    deployment.status = DeploymentStatus.STARTING.value
                    deployment.status_message = "Restarting container after system reboot..."
                    return "restarting"
                except docker.errors.APIError as e:
                    logger.error(f"Failed to restart container: {e}")
                    deployment.status = DeploymentStatus.ERROR.value
                    deployment.status_message = f"Failed to restart container: {e}"
                    return "container_missing"

            else:
                # Unknown status
                deployment.status = DeploymentStatus.STARTING.value
                deployment.status_message = f"Container status: {container_status}"
                logger.warning(
                    f"Deployment {deployment.name}: unknown container status {container_status}"
                )
                return "restarting"

        except docker.errors.NotFound:
            # Container doesn't exist
            logger.warning(f"Deployment {deployment.name}: container not found")
            deployment.status = DeploymentStatus.ERROR.value
            deployment.status_message = "Container not found after restart. Please redeploy."
            return "container_missing"

        except docker.errors.DockerException as e:
            logger.error(f"Docker error checking deployment {deployment.name}: {e}")
            deployment.status = DeploymentStatus.ERROR.value
            deployment.status_message = f"Docker error: {e}"
            return "container_missing"

    async def _check_remote_deployment(self, deployment: Deployment) -> str:
        """Check a remote deployment's status via worker API."""
        try:
            # For remote workers, check if worker is online first
            if deployment.worker.status != "online":
                deployment.status = DeploymentStatus.ERROR.value
                deployment.status_message = f"Worker {deployment.worker.name} is offline"
                logger.warning(f"Deployment {deployment.name}: worker offline")
                return "container_missing"

            # Check API health
            api_healthy = await self._check_api_health(
                deployment.worker.address,
                deployment.port,
                deployment.backend,
                None,  # No container_name for remote
            )

            if api_healthy:
                if deployment.status != DeploymentStatus.RUNNING.value:
                    deployment.status = DeploymentStatus.RUNNING.value
                    deployment.status_message = "Model ready (verified after restart)"
                logger.info(f"Deployment {deployment.name}: remote deployment healthy")
                return "running_verified"
            else:
                deployment.status = DeploymentStatus.STARTING.value
                deployment.status_message = "Waiting for model to load..."
                logger.info(f"Deployment {deployment.name}: remote API not ready")
                return "api_not_ready"

        except Exception as e:
            logger.error(f"Error checking remote deployment {deployment.name}: {e}")
            deployment.status = DeploymentStatus.ERROR.value
            deployment.status_message = f"Error checking status: {e}"
            return "container_missing"

    async def _check_api_health(
        self,
        worker_address: str,
        port: int,
        backend: str,
        container_name: str | None = None,
    ) -> bool:
        """Check if the deployment API is healthy.

        Args:
            worker_address: Worker address (host:port)
            port: Host port for the model API
            backend: Backend type (vllm, ollama, etc.)
            container_name: Container name for Docker network (local deployments)

        Returns:
            True if API is healthy, False otherwise
        """
        # Build API URL
        if container_name:
            # Local deployment - use container name on Docker network
            api_base_url = f"http://{container_name}:8000"
        else:
            # Remote deployment - use worker IP and port
            worker_ip = worker_address.split(":")[0]
            api_base_url = f"http://{worker_ip}:{port}"

        # Check /v1/models endpoint (supported by vLLM, SGLang, and Ollama)
        health_endpoint = f"{api_base_url}/v1/models"

        try:
            async with httpx.AsyncClient(timeout=self.HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(health_endpoint)

                if response.status_code == 200:
                    data = response.json()
                    # Check if models are loaded
                    if data.get("data") and len(data["data"]) > 0:
                        return True

                # For Ollama, also try native endpoint
                if backend == BackendType.OLLAMA.value:
                    ollama_endpoint = f"{api_base_url}/api/tags"
                    ollama_response = await client.get(ollama_endpoint)
                    if ollama_response.status_code == 200:
                        ollama_data = ollama_response.json()
                        if ollama_data.get("models") and len(ollama_data["models"]) > 0:
                            return True

                return False

        except httpx.ConnectError:
            logger.debug(f"API not reachable at {health_endpoint}")
            return False
        except httpx.ReadTimeout:
            logger.debug(f"API timeout at {health_endpoint}")
            return False
        except Exception as e:
            logger.debug(f"API health check error: {e}")
            return False

    async def check_deployment_health(self, deployment_id: int) -> dict:
        """Check health of a single deployment.

        Returns:
            dict with status and message
        """
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
                return {"status": "error", "message": "Deployment not found"}

            status_result = await self._check_and_update_deployment(deployment, db)
            await db.commit()

            return {
                "status": deployment.status,
                "message": deployment.status_message,
                "check_result": status_result,
            }


# Global instance
deployment_sync_service = DeploymentSyncService()

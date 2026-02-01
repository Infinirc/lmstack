"""Deployment service - handles model deployment on workers"""

import asyncio
import logging
import socket

import docker
import httpx
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.database import async_session_maker
from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import BackendType, LLMModel
from app.models.worker import OSType

logger = logging.getLogger(__name__)
settings = get_settings()


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

    # Health check configuration
    HEALTH_CHECK_INTERVAL = 5  # seconds between checks
    HEALTH_CHECK_SLOW_THRESHOLD = 600  # seconds before showing "slow loading" message (10 min)
    HEALTH_CHECK_REQUEST_TIMEOUT = 10  # timeout for each health check request

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

                # Mac with Ollama should always use native deployment (use local Ollama)
                # Mac without Docker should also use native deployment
                is_mac = worker.os_type == OSType.DARWIN.value
                is_mac_native = is_mac and (
                    backend == BackendType.OLLAMA.value or not worker.supports_docker
                )

                # Use native deployment for Mac
                if is_mac_native:
                    result = await self._deploy_native(deployment, db)
                    if result.get("error"):
                        deployment.status = DeploymentStatus.ERROR.value
                        deployment.status_message = result["error"]
                        await db.commit()
                    return

                # Build deployment request
                deploy_request = self._build_deploy_request(deployment)

                # Check if this is a local worker
                is_local = self._is_local_worker(deployment.worker.address)

                if is_local:
                    # Check if image needs to be pulled
                    image = deploy_request["image"]
                    if not self._image_exists_local(image):
                        deployment.status_message = f"Pulling image: {image}..."
                        await db.commit()

                        pull_success = await self._pull_image_local(image)
                        if not pull_success:
                            deployment.status = DeploymentStatus.ERROR.value
                            deployment.status_message = f"Failed to pull image: {image}"
                            await db.commit()
                            return

                    deployment.status_message = "Starting container..."
                    await db.commit()

                    # Deploy locally using Docker directly
                    result = await self._deploy_local(deploy_request)
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
                    ollama_ready = await self._wait_for_ollama_ready(
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

                    pull_success = await self._ollama_pull_model(
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
                api_ready = await self._wait_for_api_ready(
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

    async def _wait_for_ollama_ready(
        self,
        worker_address: str,
        port: int,
        timeout: int = 60,
        container_name: str | None = None,
    ) -> bool:
        """Wait for Ollama API to be available.

        Args:
            worker_address: Worker address (host:port)
            port: Ollama container port
            timeout: Maximum wait time in seconds
            container_name: Container name for Docker network (Windows compatibility)

        Returns:
            True if Ollama is ready, False on timeout
        """
        # Ollama is configured to use port 8000 (OLLAMA_HOST=0.0.0.0:8000)
        if container_name:
            api_url = f"http://{container_name}:8000/api/tags"
        else:
            worker_ip = worker_address.split(":")[0]
            api_url = f"http://{worker_ip}:{port}/api/tags"

        logger.info(f"Waiting for Ollama API at {api_url}")

        elapsed = 0
        check_interval = 2

        async with httpx.AsyncClient(timeout=10.0) as client:
            while elapsed < timeout:
                try:
                    response = await client.get(api_url)
                    if response.status_code == 200:
                        logger.info(f"Ollama API ready after {elapsed}s")
                        return True
                except httpx.ConnectError:
                    logger.debug(f"Ollama not ready yet ({elapsed}s)")
                except Exception as e:
                    logger.debug(f"Ollama check error: {e}")

                await asyncio.sleep(check_interval)
                elapsed += check_interval

        logger.error(f"Ollama API not ready after {timeout}s")
        return False

    async def _ollama_pull_model(
        self,
        worker_address: str,
        port: int,
        model_id: str,
        container_name: str | None = None,
    ) -> bool:
        """Pull a model using Ollama API.

        Ollama requires models to be pulled before they can be used.
        This method calls the /api/pull endpoint and waits for completion.
        """
        # Ollama is configured to use port 8000 (OLLAMA_HOST=0.0.0.0:8000)
        if container_name:
            api_url = f"http://{container_name}:8000/api/pull"
        else:
            worker_ip = worker_address.split(":")[0]
            api_url = f"http://{worker_ip}:{port}/api/pull"

        logger.info(f"Pulling Ollama model: {model_id}")

        try:
            async with httpx.AsyncClient(timeout=1800.0) as client:  # 30 min timeout
                # Ollama pull is a streaming endpoint
                async with client.stream(
                    "POST",
                    api_url,
                    json={"name": model_id, "stream": True},
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"Ollama pull failed: {response.status_code}")
                        return False

                    # Process the streaming response
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                import json

                                data = json.loads(line)
                                status = data.get("status", "")

                                # Log progress
                                if "completed" in data and "total" in data:
                                    pct = int(data["completed"] / data["total"] * 100)
                                    logger.debug(f"Ollama pull: {status} ({pct}%)")
                                elif status:
                                    logger.debug(f"Ollama pull: {status}")

                                # Check for completion
                                if status == "success":
                                    logger.info(f"Ollama model {model_id} pulled successfully")
                                    return True

                            except Exception as e:
                                logger.debug(f"Error parsing Ollama response: {e}")

                    logger.info(f"Ollama model {model_id} pull completed")
                    return True

        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {api_url}")
            return False
        except Exception as e:
            logger.error(f"Ollama pull error: {e}")
            return False

    async def _wait_for_api_ready(
        self,
        worker_address: str,
        port: int,
        deployment_id: int,
        db,
        backend: str = BackendType.VLLM.value,
        container_name: str | None = None,
    ) -> bool | None:
        """
        Poll the OpenAI API endpoint until it's ready or cancelled.

        Args:
            worker_address: Worker address (host:port)
            port: Host port for the model API
            deployment_id: Deployment ID for status updates
            db: Database session
            backend: Backend type (vllm, ollama, etc.)
            container_name: Container name for local Docker network communication.
                           If set, uses container_name:8000 instead of worker_ip:port.
                           This is needed for Windows Docker Desktop compatibility.

        Returns:
            True: API is ready
            None: Cancelled (user stopped deployment)
        """
        # For local deployments with container_name, use Docker internal networking
        # All backends (vLLM, SGLang, Ollama) are configured to use port 8000
        if container_name:
            api_base_url = f"http://{container_name}:8000"
            logger.info(f"Using Docker network for API: {api_base_url}")
        else:
            worker_ip = worker_address.split(":")[0]
            api_base_url = f"http://{worker_ip}:{port}"

        # Both vLLM and Ollama support OpenAI-compatible /v1/models endpoint
        health_endpoint = f"{api_base_url}/v1/models"

        # For Ollama, we can also check /api/tags as a fallback
        is_ollama = backend == BackendType.OLLAMA.value

        elapsed = 0
        check_count = 0
        shown_slow_message = False

        logger.info(f"Waiting for API to be ready at {health_endpoint} (backend={backend})")

        async with httpx.AsyncClient(timeout=self.HEALTH_CHECK_REQUEST_TIMEOUT) as client:
            while True:  # Wait indefinitely until ready or cancelled
                check_count += 1

                # Check if deployment was cancelled
                try:
                    result = await db.execute(
                        select(Deployment).where(Deployment.id == deployment_id)
                    )
                    deployment = result.scalar_one_or_none()
                    if deployment and deployment.status in [
                        DeploymentStatus.STOPPED.value,
                        DeploymentStatus.STOPPING.value,
                    ]:
                        logger.info(f"Deployment {deployment_id} was cancelled")
                        return None
                except Exception as e:
                    logger.debug(f"Error checking deployment status: {e}")

                try:
                    response = await client.get(health_endpoint)

                    if response.status_code == 200:
                        data = response.json()
                        # vLLM returns {"object": "list", "data": [...]}
                        # Ollama returns {"object": "list", "data": [...]} (OpenAI compat)
                        if data.get("data") and len(data["data"]) > 0:
                            logger.info(
                                f"API ready at {health_endpoint} after {elapsed}s "
                                f"({check_count} checks)"
                            )
                            return True

                    # For Ollama, also try the native /api/tags endpoint
                    if is_ollama and response.status_code != 200:
                        ollama_endpoint = f"{api_base_url}/api/tags"
                        ollama_response = await client.get(ollama_endpoint)
                        if ollama_response.status_code == 200:
                            ollama_data = ollama_response.json()
                            if ollama_data.get("models") and len(ollama_data["models"]) > 0:
                                logger.info(
                                    f"Ollama API ready at {ollama_endpoint} after {elapsed}s"
                                )
                                return True

                    logger.debug(f"Health check {check_count}: status={response.status_code}")

                except httpx.ConnectError:
                    # Container not ready yet, this is expected during startup
                    logger.debug(f"Health check {check_count}: connection refused")
                except httpx.ReadTimeout:
                    logger.debug(f"Health check {check_count}: read timeout")
                except Exception as e:
                    logger.debug(f"Health check {check_count}: {type(e).__name__}: {e}")

                # Update status message periodically
                if check_count % 6 == 0:  # Every 30 seconds
                    try:
                        result = await db.execute(
                            select(Deployment).where(Deployment.id == deployment_id)
                        )
                        deployment = result.scalar_one_or_none()
                        if deployment and deployment.status == DeploymentStatus.STARTING.value:
                            mins = elapsed // 60
                            secs = elapsed % 60
                            time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

                            # Show patience message after threshold
                            if (
                                elapsed >= self.HEALTH_CHECK_SLOW_THRESHOLD
                                and not shown_slow_message
                            ):
                                deployment.status_message = (
                                    f"Loading model... ({time_str}) - "
                                    "Large model or slow network detected. Please be patient."
                                )
                                shown_slow_message = True
                            elif shown_slow_message:
                                deployment.status_message = (
                                    f"Loading model... ({time_str}) - Please be patient."
                                )
                            else:
                                deployment.status_message = (
                                    f"Loading model into GPU memory... ({time_str})"
                                )
                            await db.commit()
                    except Exception as e:
                        logger.debug(f"Error updating deployment status message: {e}")

                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
                elapsed += self.HEALTH_CHECK_INTERVAL

    def _is_local_worker(self, address: str) -> bool:
        """Check if the worker address refers to the local machine."""
        if not address:
            return False
        host = address.split(":")[0].lower()
        return host in ("localhost", "127.0.0.1", "local")

    async def _deploy_native(self, deployment: Deployment, db) -> dict:
        """Deploy using native backend (Mac without Docker).

        Supports Ollama, MLX, and llama.cpp backends on macOS.
        """
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

            api_ready = await self._wait_for_native_api_ready(
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

    async def _wait_for_native_api_ready(
        self,
        worker_address: str,
        port: int,
        deployment_id: int,
        db,
        backend: str = BackendType.OLLAMA.value,
        timeout: int = 300,
    ) -> bool | None:
        """Wait for native API to be ready.

        For native deployments, we check via worker agent because
        Ollama only listens on localhost.

        Returns:
            True: API is ready
            False: Timeout
            None: Cancelled
        """
        # For native deployments, check via worker agent's health endpoint
        worker_health_url = f"http://{worker_address}/native/health"

        elapsed = 0
        check_interval = 5

        async with httpx.AsyncClient(timeout=10.0) as client:
            while elapsed < timeout:
                # Check if cancelled
                try:
                    result = await db.execute(
                        select(Deployment).where(Deployment.id == deployment_id)
                    )
                    deployment = result.scalar_one_or_none()
                    if deployment and deployment.status in [
                        DeploymentStatus.STOPPED.value,
                        DeploymentStatus.STOPPING.value,
                    ]:
                        return None
                except Exception:
                    pass

                try:
                    response = await client.get(
                        worker_health_url, params={"backend": backend, "port": port}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("ready"):
                            logger.info("Native API ready (checked via worker)")
                            return True
                except Exception:
                    pass

                await asyncio.sleep(check_interval)
                elapsed += check_interval

                # Update status periodically
                if elapsed % 30 == 0:
                    try:
                        result = await db.execute(
                            select(Deployment).where(Deployment.id == deployment_id)
                        )
                        deployment = result.scalar_one_or_none()
                        if deployment:
                            deployment.status_message = f"Loading model... ({elapsed}s)"
                            await db.commit()
                    except Exception:
                        pass

        return False

    def _image_exists_local(self, image: str) -> bool:
        """Check if a Docker image exists locally."""
        try:
            client = docker.from_env()
            client.images.get(image)
            return True
        except docker.errors.ImageNotFound:
            return False
        except docker.errors.APIError as e:
            logger.warning(f"Docker API error checking image {image}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking image {image}: {e}")
            return False

    async def _pull_image_local(self, image: str) -> bool:
        """Pull a Docker image locally with progress logging."""
        try:
            client = docker.from_env()
            logger.info(f"Pulling image: {image}")

            # Pull with progress
            for line in client.api.pull(image, stream=True, decode=True):
                if "status" in line:
                    status = line.get("status", "")
                    progress = line.get("progress", "")
                    if progress:
                        logger.debug(f"{status}: {progress}")

            logger.info(f"Image pulled successfully: {image}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull image {image}: {e}")
            return False

    def _find_available_port(self, start_port: int = 8001, end_port: int = 9000) -> int:
        """Find an available port on the local machine."""
        for port in range(start_port, end_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No available ports in range {start_port}-{end_port}")

    async def _deploy_local(self, deploy_request: dict) -> dict:
        """Deploy a container locally using Docker.

        This is used for local workers where we don't need to go through
        a remote worker agent.

        On Windows Docker Desktop, containers must be on the same network
        to communicate. We put model containers on the 'lmstack' network
        so the backend can reach them via container name.
        """
        try:
            client = docker.from_env()

            image = deploy_request["image"]
            command = deploy_request.get("command", [])
            environment = deploy_request.get("environment", {})
            gpu_indexes = deploy_request.get("gpu_indexes", [0])
            deployment_name = deploy_request.get("deployment_name", "lmstack-deployment")

            # Find available port (still used for external access)
            host_port = self._find_available_port()

            # Container name - used for internal Docker network communication
            container_name = f"lmstack-{deployment_name}-{deploy_request['deployment_id']}"

            # Ensure lmstack network exists (for Windows Docker Desktop compatibility)
            network_name = "lmstack_lmstack"
            try:
                client.networks.get(network_name)
            except docker.errors.NotFound:
                # Try alternative network name (depends on compose project name)
                try:
                    network_name = "lmstack"
                    client.networks.get(network_name)
                except docker.errors.NotFound:
                    # Create the network if it doesn't exist
                    logger.info(f"Creating Docker network: {network_name}")
                    client.networks.create(network_name, driver="bridge")

            # Build GPU device requests
            device_requests = [
                docker.types.DeviceRequest(
                    device_ids=[str(i) for i in gpu_indexes],
                    capabilities=[["gpu"]],
                )
            ]

            # Remove existing container with same name if exists
            try:
                existing = client.containers.get(container_name)
                existing.remove(force=True)
            except docker.errors.NotFound:
                pass

            # Run container
            # Use configurable HF cache directory from settings
            hf_cache = settings.hf_cache_dir
            container = client.containers.run(
                image=image,
                command=command,
                name=container_name,
                detach=True,
                ports={"8000/tcp": host_port},
                environment=environment,
                device_requests=device_requests,
                volumes={
                    hf_cache: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                },
                shm_size="16g",  # Required for large model inference
                restart_policy={"Name": "unless-stopped"},
                network=network_name,  # Join lmstack network for Windows compatibility
            )

            logger.info(
                f"Started local container: {container.id[:12]} "
                f"(name={container_name}) on network={network_name}, port={host_port}"
            )

            return {
                "container_id": container.id,
                "container_name": container_name,
                "port": host_port,
            }

        except docker.errors.ImageNotFound as e:
            logger.error(f"Docker image not found: {e}")
            return {"error": f"Docker image not found: {deploy_request['image']}"}
        except docker.errors.APIError as e:
            logger.error(f"Docker API error: {e}")
            return {"error": f"Docker error: {str(e)}"}
        except Exception as e:
            logger.exception(f"Error deploying locally: {e}")
            return {"error": str(e)}

    def _build_deploy_request(self, deployment: Deployment) -> dict:
        """Build the deployment request for worker agent.

        Supports multiple backends:
        - vLLM: High-throughput inference with OpenAI-compatible API
        - Ollama: Simple local LLM inference with OpenAI-compatible API
        """
        model = deployment.model

        # Determine docker image based on backend
        # Priority: deployment extra_params > model docker_image > backend default
        deployment_image = (
            deployment.extra_params.get("docker_image") if deployment.extra_params else None
        )

        backend = deployment.backend

        if deployment_image:
            image = deployment_image
        elif model.docker_image:
            image = model.docker_image
        elif backend == BackendType.VLLM.value:
            image = settings.vllm_default_image
        elif backend == BackendType.SGLANG.value:
            image = settings.sglang_default_image
        elif backend == BackendType.OLLAMA.value:
            image = settings.ollama_default_image
        else:
            logger.warning(f"Unknown backend: {backend}, defaulting to vLLM")
            image = settings.vllm_default_image

        # Build command based on backend type
        if backend == BackendType.OLLAMA.value:
            cmd, env = self._build_ollama_config(model, deployment)
        elif backend == BackendType.SGLANG.value:
            cmd, env = self._build_sglang_config(model, deployment)
        else:
            cmd, env = self._build_vllm_config(model, deployment)

        request = {
            "deployment_id": deployment.id,
            "deployment_name": deployment.name,
            "image": image,
            "command": cmd,
            "model_id": model.model_id,
            "gpu_indexes": deployment.gpu_indexes or [0],
            "environment": env,
        }

        # Note: We don't reuse existing port to avoid conflicts.
        # Worker will automatically allocate an available port.

        return request

    def _build_vllm_config(
        self,
        model: "LLMModel",
        deployment: Deployment,
    ) -> tuple[list[str], dict[str, str]]:
        """Build vLLM container command and environment."""
        cmd = [
            "--model",
            model.model_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]

        # Add default params if any
        if model.default_params:
            for key, value in model.default_params.items():
                if value is True:
                    cmd.append(f"--{key}")
                elif value is not False and value is not None:
                    cmd.extend([f"--{key}", str(value)])

        # Add extra params if any (skip special keys like docker_image, custom_args)
        if deployment.extra_params:
            skip_keys = {"docker_image", "custom_args"}
            for key, value in deployment.extra_params.items():
                if key in skip_keys:
                    continue
                if value is True:
                    cmd.append(f"--{key}")
                    # Auto-add tool-call-parser when enable-auto-tool-choice is enabled
                    if key == "enable-auto-tool-choice":
                        cmd.extend(["--tool-call-parser", "hermes"])
                elif value is not False and value is not None:
                    cmd.extend([f"--{key}", str(value)])

            # Handle custom CLI arguments
            custom_args = deployment.extra_params.get("custom_args")
            if custom_args and isinstance(custom_args, str):
                # Parse custom args: split by newlines and spaces
                for line in custom_args.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Split each line by spaces for multi-arg support
                        cmd.extend(line.split())

        env = {
            "HF_HOME": "/root/.cache/huggingface",
        }

        return cmd, env

    def _build_sglang_config(
        self,
        model: "LLMModel",
        deployment: Deployment,
    ) -> tuple[list[str], dict[str, str]]:
        """Build SGLang container command and environment.

        SGLang uses similar command-line arguments to vLLM but with some
        differences in parameter names. Unlike vLLM, the sglang Docker image
        does not have a proper ENTRYPOINT, so we need to explicitly specify
        the launch command.
        """
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model.model_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]

        # Add default params if any
        if model.default_params:
            for key, value in model.default_params.items():
                if value is True:
                    cmd.append(f"--{key}")
                elif value is not False and value is not None:
                    cmd.extend([f"--{key}", str(value)])

        # Add extra params if any (skip special keys like docker_image, custom_args)
        if deployment.extra_params:
            skip_keys = {"docker_image", "custom_args"}
            for key, value in deployment.extra_params.items():
                if key in skip_keys:
                    continue
                if value is True:
                    cmd.append(f"--{key}")
                elif value is not False and value is not None:
                    cmd.extend([f"--{key}", str(value)])

            # Handle custom CLI arguments
            custom_args = deployment.extra_params.get("custom_args")
            if custom_args and isinstance(custom_args, str):
                # Parse custom args: split by newlines and spaces
                for line in custom_args.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Split each line by spaces for multi-arg support
                        cmd.extend(line.split())

        env = {
            "HF_HOME": "/root/.cache/huggingface",
        }

        return cmd, env

    def _build_ollama_config(
        self,
        model: "LLMModel",
        deployment: Deployment,
    ) -> tuple[list[str], dict[str, str]]:
        """Build Ollama container command and environment.

        Ollama uses environment variables for configuration instead of
        command-line arguments. The model is pulled and run via API after
        the container starts.
        """
        # Ollama's default entrypoint is "ollama serve"
        cmd = ["serve"]

        # Ollama environment configuration
        env = {
            "OLLAMA_HOST": "0.0.0.0:8000",  # Bind to container port 8000
            "OLLAMA_ORIGINS": "*",  # Allow CORS from all origins (required for web UI)
            "OLLAMA_NUM_PARALLEL": str(
                deployment.extra_params.get("num_parallel", 4) if deployment.extra_params else "4"
            ),
            "OLLAMA_MAX_LOADED_MODELS": str(
                deployment.extra_params.get("max_loaded_models", 1)
                if deployment.extra_params
                else "1"
            ),
            # GPU settings
            "OLLAMA_GPU_OVERHEAD": "0",
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in (deployment.gpu_indexes or [0])),
        }

        # Add any custom environment variables from extra_params
        if deployment.extra_params:
            for key, value in deployment.extra_params.items():
                if key.startswith("OLLAMA_") and value is not None:
                    env[key] = str(value)

            # Handle custom environment variables from custom_args
            custom_args = deployment.extra_params.get("custom_args")
            if custom_args and isinstance(custom_args, str):
                # Parse custom args as environment variables (KEY=VALUE format)
                for line in custom_args.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        env[key.strip()] = value.strip()

        return cmd, env

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
                        await self._stop_local(deployment.container_id)
                    else:
                        worker_url = f"http://{worker.address}/stop"

                        async with httpx.AsyncClient(timeout=60.0) as client:
                            await client.post(
                                worker_url, json={"container_id": deployment.container_id}
                            )

            except Exception as e:
                logger.warning(f"Error stopping deployment {deployment_id}: {e}")

    async def _stop_local(self, container_id: str) -> None:
        """Stop a container locally."""
        try:
            client = docker.from_env()
            container = client.containers.get(container_id)
            container.stop(timeout=30)
            container.remove()
            logger.info(f"Stopped local container: {container_id[:12]}")
        except docker.errors.NotFound:
            logger.warning(f"Container not found: {container_id}")
        except Exception as e:
            logger.warning(f"Error stopping local container: {e}")

    async def get_logs(self, deployment: Deployment, tail: int = 100) -> str:
        """Get logs from a deployment"""
        if not deployment.container_id or not deployment.worker:
            return "No container running"

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
                    return self._get_logs_local(deployment.container_id, tail)
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

    def _get_logs_local(self, container_id: str, tail: int = 100) -> str:
        """Get logs from a local container."""
        try:
            client = docker.from_env()
            container = client.containers.get(container_id)
            logs = container.logs(tail=tail, timestamps=True).decode("utf-8")
            return logs
        except docker.errors.NotFound:
            return "Container not found"
        except Exception as e:
            return f"Error: {str(e)}"

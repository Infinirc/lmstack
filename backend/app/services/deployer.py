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

logger = logging.getLogger(__name__)
settings = get_settings()


class DeployerService:
    """Service for deploying models to workers"""

    # Health check configuration
    HEALTH_CHECK_INTERVAL = 5  # seconds between checks
    HEALTH_CHECK_TIMEOUT = 600  # max seconds to wait (10 minutes for large models)
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
                else:
                    # Send to remote worker agent
                    worker_url = f"http://{deployment.worker.address}/deploy"

                    async with httpx.AsyncClient(timeout=300.0) as client:
                        response = await client.post(worker_url, json=deploy_request)

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
                    deployment.status_message = "Pulling model with Ollama..."
                    await db.commit()

                    pull_success = await self._ollama_pull_model(
                        deployment.worker.address,
                        deployment.port,
                        deployment.model.model_id,
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
                )

                # Refresh deployment object after health check updates
                await db.refresh(deployment)

                if api_ready is None:
                    # Deployment was cancelled, don't update status
                    logger.info(f"Deployment {deployment_id} cancelled during startup")
                    return
                elif api_ready:
                    deployment.status = DeploymentStatus.RUNNING.value
                    deployment.status_message = "Model ready"
                else:
                    deployment.status = DeploymentStatus.ERROR.value
                    deployment.status_message = "Model failed to start within timeout"

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

    async def _ollama_pull_model(
        self,
        worker_address: str,
        port: int,
        model_id: str,
    ) -> bool:
        """Pull a model using Ollama API.

        Ollama requires models to be pulled before they can be used.
        This method calls the /api/pull endpoint and waits for completion.
        """
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
    ) -> bool | None:
        """
        Poll the OpenAI API endpoint until it's ready or timeout.

        Returns:
            True: API is ready
            False: Timeout or error
            None: Cancelled (user stopped deployment)
        """
        worker_ip = worker_address.split(":")[0]
        api_base_url = f"http://{worker_ip}:{port}"

        # Both vLLM and Ollama support OpenAI-compatible /v1/models endpoint
        health_endpoint = f"{api_base_url}/v1/models"

        # For Ollama, we can also check /api/tags as a fallback
        is_ollama = backend == BackendType.OLLAMA.value

        elapsed = 0
        check_count = 0

        logger.info(f"Waiting for API to be ready at {health_endpoint} (backend={backend})")

        async with httpx.AsyncClient(timeout=self.HEALTH_CHECK_REQUEST_TIMEOUT) as client:
            while elapsed < self.HEALTH_CHECK_TIMEOUT:
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
                            deployment.status_message = (
                                f"Loading model into GPU memory... ({time_str})"
                            )
                            await db.commit()
                    except Exception as e:
                        logger.debug(f"Error updating deployment status message: {e}")

                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
                elapsed += self.HEALTH_CHECK_INTERVAL

        logger.warning(f"API health check timed out after {elapsed}s ({check_count} checks)")
        return False

    def _is_local_worker(self, address: str) -> bool:
        """Check if the worker address refers to the local machine."""
        if not address:
            return False
        host = address.split(":")[0].lower()
        return host in ("localhost", "127.0.0.1", "local")

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
        """
        try:
            client = docker.from_env()

            image = deploy_request["image"]
            command = deploy_request.get("command", [])
            environment = deploy_request.get("environment", {})
            gpu_indexes = deploy_request.get("gpu_indexes", [0])
            deployment_name = deploy_request.get("deployment_name", "lmstack-deployment")

            # Find available port
            host_port = self._find_available_port()

            # Container name
            container_name = f"lmstack-{deployment_name}-{deploy_request['deployment_id']}"

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
            )

            logger.info(f"Started local container: {container.id[:12]} on port {host_port}")

            return {
                "container_id": container.id,
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
        differences in parameter names.
        """
        cmd = [
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
                is_local = self._is_local_worker(deployment.worker.address)

                if is_local:
                    # Stop locally using Docker directly
                    await self._stop_local(deployment.container_id)
                else:
                    worker_url = f"http://{deployment.worker.address}/stop"

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
            is_local = self._is_local_worker(deployment.worker.address)

            if is_local:
                return self._get_logs_local(deployment.container_id, tail)
            else:
                worker_url = f"http://{deployment.worker.address}/logs"

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

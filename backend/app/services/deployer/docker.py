"""Docker deployment operations.

This module handles Docker-specific deployment operations including
image management, container lifecycle, and local Docker operations.
"""

import logging
import socket

import docker

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def image_exists_local(image: str) -> bool:
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


async def pull_image_local(image: str) -> bool:
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


def find_available_port(start_port: int = 8001, end_port: int = 9000) -> int:
    """Find an available port on the local machine."""
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available ports in range {start_port}-{end_port}")


async def deploy_local(deploy_request: dict) -> dict:
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
        host_port = find_available_port()

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


async def stop_local(container_id: str) -> None:
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


def get_logs_local(container_id: str, tail: int = 100) -> str:
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

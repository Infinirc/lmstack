"""Docker runner for LMStack model deployments.

Provides container management specifically for LLM model inference containers.
"""

import asyncio
import logging
import socket
from typing import Optional

import docker
from docker.errors import APIError, NotFound

logger = logging.getLogger(__name__)


# =============================================================================
# Progress Tracking
# =============================================================================

# In-memory store for tracking image pull progress.
# NOTE: For production with multiple workers, use Redis or database storage.
_pull_progress: dict[str, dict] = {}
_MAX_PROGRESS_ENTRIES = 100


def get_pull_progress(deployment_id: int) -> dict:
    """Get pull progress for a deployment."""
    return _pull_progress.get(str(deployment_id), {"status": "unknown"})


def _set_pull_progress(deployment_id: int, data: dict) -> None:
    """Set pull progress for a deployment."""
    _pull_progress[str(deployment_id)] = data


def _cleanup_old_progress() -> None:
    """Remove old progress entries to prevent memory leaks."""
    if len(_pull_progress) > _MAX_PROGRESS_ENTRIES:
        # Remove completed or errored entries first
        completed_keys = [
            key
            for key, val in _pull_progress.items()
            if val.get("status") in ("completed", "error")
        ]
        for key in completed_keys[: len(_pull_progress) - _MAX_PROGRESS_ENTRIES // 2]:
            _pull_progress.pop(key, None)


# =============================================================================
# Docker Runner
# =============================================================================


class DockerRunner:
    """Manage Docker containers for model inference."""

    PORT_RANGE_START = 40000
    PORT_RANGE_END = 49999

    def __init__(self):
        self.client = docker.from_env()
        self._port_counter = self.PORT_RANGE_START

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use by trying to bind to it."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return False
            except OSError:
                return True

    def _get_docker_used_ports(self) -> set[int]:
        """Get all ports currently used by Docker containers."""
        used_ports: set[int] = set()
        try:
            containers = self.client.containers.list(all=False)  # Only running
            for container in containers:
                ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
                if ports:
                    for bindings in ports.values():
                        if bindings:
                            for binding in bindings:
                                host_port = binding.get("HostPort")
                                if host_port:
                                    used_ports.add(int(host_port))
        except docker.errors.DockerException as e:
            logger.warning(f"Failed to get Docker ports: {e}")
        return used_ports

    def _get_next_port(self) -> int:
        """Get the next available port that is not in use."""
        docker_ports = self._get_docker_used_ports()
        attempts = 0
        max_attempts = self.PORT_RANGE_END - self.PORT_RANGE_START + 1

        while attempts < max_attempts:
            port = self._port_counter
            self._port_counter += 1
            if self._port_counter > self.PORT_RANGE_END:
                self._port_counter = self.PORT_RANGE_START

            # Skip if port is used by Docker or system
            if port in docker_ports:
                attempts += 1
                continue

            if not self._is_port_in_use(port):
                return port

            attempts += 1

        raise RuntimeError(
            f"No available port found in range {self.PORT_RANGE_START}-{self.PORT_RANGE_END}"
        )

    def pull_image_with_progress(self, image: str, deployment_id: int) -> None:
        """Pull image with progress tracking."""
        _cleanup_old_progress()

        _set_pull_progress(
            deployment_id,
            {
                "status": "pulling",
                "image": image,
                "progress": 0,
                "layers": {},
            },
        )

        try:
            for line in self.client.api.pull(image, stream=True, decode=True):
                if "id" in line and "progressDetail" in line:
                    layer_id = line["id"]
                    detail = line.get("progressDetail", {})
                    current = detail.get("current", 0)
                    total = detail.get("total", 0)

                    progress_data = _pull_progress.get(str(deployment_id), {})
                    layers = progress_data.get("layers", {})
                    layers[layer_id] = {
                        "status": line.get("status", ""),
                        "current": current,
                        "total": total,
                    }

                    # Calculate overall progress
                    total_size = sum(layer.get("total", 0) for layer in layers.values())
                    downloaded = sum(layer.get("current", 0) for layer in layers.values())
                    overall_progress = int((downloaded / total_size) * 100) if total_size > 0 else 0

                    _set_pull_progress(
                        deployment_id,
                        {
                            "status": "pulling",
                            "image": image,
                            "progress": overall_progress,
                            "layers": layers,
                        },
                    )

                elif "status" in line:
                    logger.info(f"Pull: {line.get('status')}")

            _set_pull_progress(
                deployment_id,
                {
                    "status": "completed",
                    "image": image,
                    "progress": 100,
                    "layers": _pull_progress.get(str(deployment_id), {}).get("layers", {}),
                },
            )

        except APIError as e:
            _set_pull_progress(
                deployment_id,
                {
                    "status": "error",
                    "image": image,
                    "error": str(e),
                },
            )
            raise

    async def run(
        self,
        name: str,
        image: str,
        command: list[str],
        gpu_indexes: list[int],
        environment: dict[str, str],
        deployment_id: int = 0,
        port: Optional[int] = None,
    ) -> tuple[str, int]:
        """Run a container and return (container_id, port)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._run_sync,
            name,
            image,
            command,
            gpu_indexes,
            environment,
            deployment_id,
            port,
        )

    def _run_sync(
        self,
        name: str,
        image: str,
        command: list[str],
        gpu_indexes: list[int],
        environment: dict[str, str],
        deployment_id: int = 0,
        port: Optional[int] = None,
    ) -> tuple[str, int]:
        """Synchronous container run."""
        # Check if container with same name exists
        try:
            existing = self.client.containers.get(name)
            logger.info(f"Removing existing container {name}")
            existing.remove(force=True)
        except NotFound:
            pass

        # Pull image if not exists
        try:
            self.client.images.get(image)
        except NotFound:
            logger.info(f"Pulling image {image}...")
            self.pull_image_with_progress(image, deployment_id)

        # Use specified port or get a new one
        host_port = port if port else self._get_next_port()

        # Build GPU device request
        gpu_device_ids = [str(i) for i in gpu_indexes]

        device_requests = [
            docker.types.DeviceRequest(
                device_ids=gpu_device_ids,
                capabilities=[["gpu"]],
            )
        ]

        # Merge environment
        env = {
            "NVIDIA_VISIBLE_DEVICES": ",".join(gpu_device_ids),
            **environment,
        }

        # Run container
        container = self.client.containers.run(
            image=image,
            name=name,
            command=command,
            detach=True,
            remove=False,
            ports={"8000/tcp": host_port},
            device_requests=device_requests,
            environment=env,
            shm_size="16g",
            volumes={
                "/root/.cache/huggingface": {
                    "bind": "/root/.cache/huggingface",
                    "mode": "rw",
                },
            },
        )

        return container.id, host_port

    async def stop(self, container_id: str) -> None:
        """Stop and remove a container."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._stop_sync, container_id)

    def _stop_sync(self, container_id: str) -> None:
        """Synchronous container stop."""
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=30)
            container.remove()
            logger.info(f"Container {container_id[:12]} stopped and removed")
        except NotFound:
            logger.warning(f"Container {container_id[:12]} not found")
        except APIError as e:
            logger.error(f"Failed to stop container: {e}")
            raise

    async def logs(self, container_id: str, tail: int = 100) -> str:
        """Get container logs."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._logs_sync, container_id, tail)

    def _logs_sync(self, container_id: str, tail: int = 100) -> str:
        """Synchronous log retrieval."""
        try:
            container = self.client.containers.get(container_id)
            logs = container.logs(tail=tail, timestamps=True)
            return logs.decode("utf-8", errors="replace")
        except NotFound:
            return "Container not found"
        except APIError as e:
            return f"Error: {str(e)}"

    def list_containers(self, prefix: str = "lmstack-") -> list[dict]:
        """List all LMStack containers."""
        containers = []

        for container in self.client.containers.list(all=True):
            if container.name.startswith(prefix):
                containers.append(
                    {
                        "id": container.id,
                        "name": container.name,
                        "status": container.status,
                        "image": (container.image.tags[0] if container.image.tags else "unknown"),
                    }
                )

        return containers

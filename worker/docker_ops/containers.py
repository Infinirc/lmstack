"""Docker container management for LMStack Worker.

Provides methods for listing, inspecting, creating, starting,
stopping, and removing Docker containers on the worker node.
"""

import logging
from typing import Any, Optional

import docker
from docker.errors import APIError, NotFound

logger = logging.getLogger(__name__)


class ContainerManager:
    """Docker container management operations."""

    # LMStack container label key
    LMSTACK_LABEL = "lmstack.managed"
    DEPLOYMENT_LABEL = "lmstack.deployment_id"

    def __init__(self, client: docker.DockerClient):
        self.client = client

    def list_containers(
        self,
        all: bool = True,
        managed_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List containers on this node.

        Args:
            all: Include stopped containers
            managed_only: Only return LMStack-managed containers

        Returns:
            List of container dictionaries
        """
        containers = []

        for container in self.client.containers.list(all=all):
            # Check if managed by LMStack
            labels = container.labels or {}
            is_managed = labels.get(self.LMSTACK_LABEL) == "true" or container.name.startswith(
                "lmstack-"
            )

            if managed_only and not is_managed:
                continue

            # Get deployment info if available
            deployment_id = None
            deployment_name = None
            if is_managed:
                deployment_id_str = labels.get(self.DEPLOYMENT_LABEL)
                if deployment_id_str:
                    try:
                        deployment_id = int(deployment_id_str)
                    except ValueError:
                        pass
                # Extract deployment name from container name
                if container.name.startswith("lmstack-"):
                    deployment_name = container.name[8:]  # Remove "lmstack-" prefix

            # Parse ports
            ports = self._parse_ports(container)

            # Parse volumes/mounts
            volumes = self._parse_volumes(container)

            # Get GPU info from environment
            gpu_ids = self._get_gpu_ids(container)

            # Get image info
            image_tags = container.image.tags if container.image else []
            image_name = (
                image_tags[0]
                if image_tags
                else container.attrs.get("Config", {}).get("Image", "unknown")
            )

            # Parse timestamps
            state = container.attrs.get("State", {})
            started_at = state.get("StartedAt")
            finished_at = state.get("FinishedAt")

            containers.append(
                {
                    "id": container.short_id,
                    "name": container.name,
                    "image": image_name,
                    "image_id": container.image.short_id if container.image else "",
                    "state": container.status,
                    "status": container.attrs.get("Status", container.status),
                    "created_at": self._normalize_timestamp(container.attrs.get("Created")),
                    "started_at": self._normalize_timestamp(started_at),
                    "finished_at": self._normalize_timestamp(finished_at),
                    "exit_code": state.get("ExitCode"),
                    "ports": ports,
                    "volumes": volumes,
                    "gpu_ids": gpu_ids,
                    "deployment_id": deployment_id,
                    "deployment_name": deployment_name,
                    "is_managed": is_managed,
                }
            )

        return containers

    def _normalize_timestamp(self, ts: Optional[str]) -> Optional[str]:
        """Normalize timestamp format (remove nanoseconds)."""
        if not ts or ts == "0001-01-01T00:00:00Z":
            return None
        if "." in ts:
            return ts.split(".")[0] + "Z"
        return ts

    def _parse_ports(self, container) -> list[dict]:
        """Parse port bindings from container."""
        ports = []
        port_bindings = container.attrs.get("NetworkSettings", {}).get("Ports", {})
        for container_port, bindings in (port_bindings or {}).items():
            if bindings:
                for binding in bindings:
                    port_str, protocol = container_port.split("/")
                    ports.append(
                        {
                            "container_port": int(port_str),
                            "host_port": int(binding.get("HostPort", 0)),
                            "protocol": protocol,
                            "host_ip": binding.get("HostIp", "0.0.0.0"),
                        }
                    )
        return ports

    def _parse_volumes(self, container) -> list[dict]:
        """Parse volume mounts from container."""
        volumes = []
        for mount in container.attrs.get("Mounts", []):
            volumes.append(
                {
                    "source": mount.get("Source", ""),
                    "destination": mount.get("Destination", ""),
                    "mode": "ro" if mount.get("RW") is False else "rw",
                    "type": mount.get("Type", "bind"),
                }
            )
        return volumes

    def _get_gpu_ids(self, container) -> Optional[list[str]]:
        """Get GPU IDs from container environment."""
        env_list = container.attrs.get("Config", {}).get("Env", [])
        for env in env_list:
            if env.startswith("NVIDIA_VISIBLE_DEVICES="):
                gpu_str = env.split("=", 1)[1]
                if gpu_str and gpu_str != "all":
                    return gpu_str.split(",")
        return None

    def get_container_detail(self, container_id: str) -> dict[str, Any]:
        """Get detailed container information.

        Args:
            container_id: Container ID or name

        Returns:
            Detailed container information
        """
        container = self.client.containers.get(container_id)

        # Get basic info using list format
        containers = self.list_containers(all=True)
        for c in containers:
            if c["id"] == container.short_id or c["name"] == container_id:
                # Add environment variables (mask sensitive values)
                c["env"] = self._mask_sensitive_env(container)
                return c

        raise NotFound(f"Container {container_id} not found")

    def _mask_sensitive_env(self, container) -> list[str]:
        """Mask sensitive environment variables."""
        env_list = container.attrs.get("Config", {}).get("Env", [])
        masked_env = []
        sensitive_keys = {"password", "secret", "token", "key", "auth"}

        for env in env_list:
            if "=" in env:
                key, value = env.split("=", 1)
                if any(s in key.lower() for s in sensitive_keys):
                    masked_env.append(f"{key}=***")
                else:
                    masked_env.append(env)
            else:
                masked_env.append(env)

        return masked_env

    def get_container_stats(self, container_id: str) -> dict[str, Any]:
        """Get container resource usage stats.

        Args:
            container_id: Container ID or name

        Returns:
            Container resource usage statistics
        """
        container = self.client.containers.get(container_id)
        stats = container.stats(stream=False)

        # Calculate CPU percentage
        cpu_delta = (
            stats["cpu_stats"]["cpu_usage"]["total_usage"]
            - stats["precpu_stats"]["cpu_usage"]["total_usage"]
        )
        system_delta = (
            stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
        )
        num_cpus = stats["cpu_stats"]["online_cpus"]

        cpu_percent = 0.0
        if system_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0

        # Memory stats
        mem_usage = stats["memory_stats"].get("usage", 0)
        mem_limit = stats["memory_stats"].get("limit", 0)
        mem_percent = (mem_usage / mem_limit * 100) if mem_limit > 0 else 0

        # Network stats
        networks = stats.get("networks", {})
        network_rx = sum(n.get("rx_bytes", 0) for n in networks.values())
        network_tx = sum(n.get("tx_bytes", 0) for n in networks.values())

        # Block I/O stats
        blkio = stats.get("blkio_stats", {}).get("io_service_bytes_recursive", [])
        block_read = sum(b.get("value", 0) for b in (blkio or []) if b.get("op") == "read")
        block_write = sum(b.get("value", 0) for b in (blkio or []) if b.get("op") == "write")

        return {
            "cpu_percent": round(cpu_percent, 2),
            "memory_usage": mem_usage,
            "memory_limit": mem_limit,
            "memory_percent": round(mem_percent, 2),
            "network_rx": network_rx,
            "network_tx": network_tx,
            "block_read": block_read,
            "block_write": block_write,
            "pids": stats.get("pids_stats", {}).get("current", 0),
        }

    def create_container(
        self,
        name: str,
        image: str,
        command: Optional[list[str]] = None,
        entrypoint: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        ports: Optional[list[dict]] = None,
        volumes: Optional[list[dict]] = None,
        gpu_ids: Optional[list[int]] = None,
        restart_policy: str = "no",
        labels: Optional[dict[str, str]] = None,
        cpu_limit: Optional[float] = None,
        memory_limit: Optional[int] = None,
        cap_add: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Create and start a new container.

        Args:
            name: Container name
            image: Image to use
            command: Command to run
            entrypoint: Override entrypoint
            env: Environment variables
            ports: Port mappings
            volumes: Volume mounts
            gpu_ids: GPU device IDs to use
            restart_policy: Restart policy
            labels: Container labels
            cpu_limit: CPU limit (number of CPUs)
            memory_limit: Memory limit in bytes
            cap_add: Linux capabilities to add (e.g., ["SYS_ADMIN"])

        Returns:
            Created container information
        """
        logger.info(f"Creating container: {name} from image {image}")

        # Remove existing container with same name
        try:
            existing = self.client.containers.get(name)
            logger.info(f"Removing existing container {name}")
            existing.remove(force=True)
        except NotFound:
            pass

        # Build port bindings
        port_bindings = {}
        if ports:
            for p in ports:
                container_port = f"{p['container_port']}/{p.get('protocol', 'tcp')}"
                port_bindings[container_port] = p.get("host_port")

        # Build volume mounts
        volume_bindings = {}
        if volumes:
            for v in volumes:
                volume_bindings[v["source"]] = {
                    "bind": v["destination"],
                    "mode": v.get("mode", "rw"),
                }

        # Build GPU device requests
        device_requests = None
        environment = dict(env) if env else {}
        if gpu_ids is not None:
            gpu_device_ids = [str(i) for i in gpu_ids]
            device_requests = [
                docker.types.DeviceRequest(
                    device_ids=gpu_device_ids,
                    capabilities=[["gpu"]],
                )
            ]
            environment["NVIDIA_VISIBLE_DEVICES"] = ",".join(gpu_device_ids)

        # Build labels
        container_labels = dict(labels) if labels else {}
        container_labels[self.LMSTACK_LABEL] = "true"

        # Build restart policy
        restart_config = {"Name": restart_policy}
        if restart_policy == "on-failure":
            restart_config["MaximumRetryCount"] = 3

        container = self.client.containers.run(
            image=image,
            name=name,
            command=command,
            entrypoint=entrypoint,
            detach=True,
            remove=False,
            ports=port_bindings if port_bindings else None,
            volumes=volume_bindings if volume_bindings else None,
            device_requests=device_requests,
            environment=environment,
            labels=container_labels,
            restart_policy=restart_config,
            cpu_period=100000 if cpu_limit else None,
            cpu_quota=int(cpu_limit * 100000) if cpu_limit else None,
            mem_limit=memory_limit,
            cap_add=cap_add,
        )

        logger.info(f"Container {name} created with ID {container.short_id}")
        return self.get_container_detail(container.id)

    def start_container(self, container_id: str) -> dict[str, Any]:
        """Start a stopped container."""
        logger.info(f"Starting container: {container_id}")
        container = self.client.containers.get(container_id)
        container.start()
        container.reload()
        return self.get_container_detail(container.id)

    def stop_container(self, container_id: str, timeout: int = 10) -> dict[str, Any]:
        """Stop a running container."""
        logger.info(f"Stopping container: {container_id}")
        container = self.client.containers.get(container_id)
        container.stop(timeout=timeout)
        container.reload()
        return self.get_container_detail(container.id)

    def restart_container(self, container_id: str, timeout: int = 10) -> dict[str, Any]:
        """Restart a container."""
        logger.info(f"Restarting container: {container_id}")
        container = self.client.containers.get(container_id)
        container.restart(timeout=timeout)
        container.reload()
        return self.get_container_detail(container.id)

    def pause_container(self, container_id: str) -> dict[str, Any]:
        """Pause a running container."""
        logger.info(f"Pausing container: {container_id}")
        container = self.client.containers.get(container_id)
        container.pause()
        container.reload()
        return self.get_container_detail(container.id)

    def unpause_container(self, container_id: str) -> dict[str, Any]:
        """Unpause a paused container."""
        logger.info(f"Unpausing container: {container_id}")
        container = self.client.containers.get(container_id)
        container.unpause()
        container.reload()
        return self.get_container_detail(container.id)

    def remove_container(
        self,
        container_id: str,
        force: bool = False,
        remove_volumes: bool = False,
    ) -> None:
        """Remove a container.

        Args:
            container_id: Container ID or name
            force: Force removal of running container
            remove_volumes: Remove associated volumes
        """
        logger.info(f"Removing container: {container_id}")
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=force, v=remove_volumes)
        except NotFound:
            logger.warning(f"Container {container_id} not found")
            raise
        except APIError as e:
            logger.error(f"Failed to remove container {container_id}: {e}")
            raise

    def get_logs(
        self,
        container_id: str,
        tail: int = 100,
        since: Optional[int] = None,
        until: Optional[int] = None,
        timestamps: bool = True,
    ) -> dict[str, str]:
        """Get container logs.

        Args:
            container_id: Container ID or name
            tail: Number of lines from the end
            since: Unix timestamp to start from
            until: Unix timestamp to end at
            timestamps: Include timestamps

        Returns:
            Log output with stdout and stderr
        """
        container = self.client.containers.get(container_id)

        # Get logs with separate streams
        stdout_logs = container.logs(
            stdout=True,
            stderr=False,
            tail=tail,
            since=since,
            until=until,
            timestamps=timestamps,
        ).decode("utf-8", errors="replace")

        stderr_logs = container.logs(
            stdout=False,
            stderr=True,
            tail=tail,
            since=since,
            until=until,
            timestamps=timestamps,
        ).decode("utf-8", errors="replace")

        # Combined logs
        combined_logs = container.logs(
            stdout=True,
            stderr=True,
            tail=tail,
            since=since,
            until=until,
            timestamps=timestamps,
        ).decode("utf-8", errors="replace")

        return {
            "container_id": container.short_id,
            "logs": combined_logs,
            "stdout": stdout_logs,
            "stderr": stderr_logs,
        }

    def exec_command(
        self,
        container_id: str,
        command: list[str],
        tty: bool = False,
        privileged: bool = False,
        user: Optional[str] = None,
        workdir: Optional[str] = None,
        env: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Execute a command in a running container.

        Args:
            container_id: Container ID or name
            command: Command to execute
            tty: Allocate a pseudo-TTY
            privileged: Run in privileged mode
            user: User to run as
            workdir: Working directory
            env: Environment variables

        Returns:
            Command execution result
        """
        logger.info(f"Executing command in container {container_id}: {' '.join(command)}")
        container = self.client.containers.get(container_id)

        exit_code, output = container.exec_run(
            cmd=command,
            tty=tty,
            privileged=privileged,
            user=user,
            workdir=workdir,
            environment=env,
            demux=True,  # Separate stdout and stderr
        )

        stdout = output[0].decode("utf-8", errors="replace") if output[0] else ""
        stderr = output[1].decode("utf-8", errors="replace") if output[1] else ""

        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }

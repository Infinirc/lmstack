"""Headscale Manager

Manages Headscale VPN server for connecting remote workers.
Headscale is a self-hosted implementation of the Tailscale control server.
"""
import asyncio
import json
import logging
import os
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import docker
from docker.errors import NotFound, APIError
import httpx

logger = logging.getLogger(__name__)

HEADSCALE_CONTAINER_NAME = "lmstack-headscale"
HEADSCALE_IMAGE = "headscale/headscale:latest"
# Use user home directory to avoid permission issues
HEADSCALE_DATA_DIR = os.path.expanduser("~/.lmstack/headscale")
HEADSCALE_CONFIG_PATH = f"{HEADSCALE_DATA_DIR}/config.yaml"
HEADSCALE_DB_PATH = f"{HEADSCALE_DATA_DIR}/db.sqlite"
HEADSCALE_SOCKET_PATH = f"{HEADSCALE_DATA_DIR}/headscale.sock"

# Default ports (use 8090 to avoid conflict with common services)
HEADSCALE_HTTP_PORT = 8090
HEADSCALE_GRPC_PORT = 50443

# LMStack user in Headscale
LMSTACK_USER = "lmstack"


def _parse_timestamp(value) -> Optional[str]:
    """Parse timestamp from various formats to ISO string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # New Headscale format: {seconds: ..., nanos: ...}
        seconds = value.get("seconds", 0)
        if seconds and seconds > 0:
            return datetime.fromtimestamp(seconds).isoformat()
    return None


class HeadscaleNode:
    """Represents a node in Headscale."""
    def __init__(self, data: dict):
        self.id = data.get("id")
        self.name = data.get("name") or data.get("givenName") or data.get("given_name")
        self.given_name = data.get("givenName") or data.get("given_name")
        # Support both camelCase and snake_case (newer Headscale uses snake_case)
        self.ip_addresses = data.get("ipAddresses") or data.get("ip_addresses", [])
        self.online = data.get("online", False)
        self.last_seen = _parse_timestamp(data.get("lastSeen") or data.get("last_seen"))
        self.created_at = _parse_timestamp(data.get("createdAt") or data.get("created_at"))

    @property
    def ipv4(self) -> Optional[str]:
        """Get the first IPv4 address."""
        for ip in self.ip_addresses:
            if "." in ip:
                return ip
        return None


class HeadscaleManager:
    """Manages Headscale VPN server."""

    def __init__(self, server_url: str = "http://localhost", http_port: int = HEADSCALE_HTTP_PORT):
        self._client: Optional[docker.DockerClient] = None
        self.server_url = server_url
        self.http_port = http_port
        self._api_key: Optional[str] = None
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(HEADSCALE_DATA_DIR, exist_ok=True)
        os.makedirs(f"{HEADSCALE_DATA_DIR}/run", exist_ok=True)

    @property
    def client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    @property
    def api_base_url(self) -> str:
        """Get Headscale API base URL."""
        return f"http://localhost:{self.http_port}"

    def _get_container(self) -> Optional[docker.models.containers.Container]:
        """Get Headscale container if exists."""
        try:
            return self.client.containers.get(HEADSCALE_CONTAINER_NAME)
        except NotFound:
            return None

    def _generate_config(self, server_url: str, http_port: int, grpc_port: int) -> dict:
        """Generate Headscale configuration."""
        return {
            "server_url": f"{server_url}:{http_port}",
            "listen_addr": f"0.0.0.0:{http_port}",
            "grpc_listen_addr": f"0.0.0.0:{grpc_port}",
            "grpc_allow_insecure": True,
            "private_key_path": "/etc/headscale/private.key",
            "noise": {
                "private_key_path": "/etc/headscale/noise_private.key"
            },
            "prefixes": {
                "v4": "100.64.0.0/10",
                "v6": "fd7a:115c:a1e0::/48"
            },
            "derp": {
                "server": {
                    "enabled": False
                },
                "urls": ["https://controlplane.tailscale.com/derpmap/default"],
                "auto_update_enabled": True,
                "update_frequency": "24h"
            },
            "disable_check_updates": True,
            "ephemeral_node_inactivity_timeout": "30m",
            "database": {
                "type": "sqlite3",
                "sqlite": {
                    "path": "/etc/headscale/db.sqlite"
                }
            },
            "log": {
                "format": "text",
                "level": "info"
            },
            "dns": {
                "magic_dns": True,
                "base_domain": "lmstack.local",
                "nameservers": {
                    "global": ["1.1.1.1", "8.8.8.8"]
                }
            },
            "unix_socket": "/var/run/headscale/headscale.sock",
            "unix_socket_permission": "0770"
        }

    def _write_config(self, server_url: str, http_port: int, grpc_port: int):
        """Write Headscale configuration file."""
        config = self._generate_config(server_url, http_port, grpc_port)
        with open(HEADSCALE_CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Headscale config written to {HEADSCALE_CONFIG_PATH}")

    async def is_running(self) -> bool:
        """Check if Headscale is running."""
        container = self._get_container()
        return container is not None and container.status == "running"

    async def start(
        self,
        server_url: str,
        http_port: int = HEADSCALE_HTTP_PORT,
        grpc_port: int = HEADSCALE_GRPC_PORT,
    ) -> bool:
        """Start Headscale server."""
        self.server_url = server_url
        self.http_port = http_port

        container = self._get_container()

        if container:
            if container.status == "running":
                logger.info("Headscale is already running")
                return True
            # Remove old container to ensure port config is correct
            logger.info("Removing old Headscale container")
            container.remove(force=True)

        # Write config
        self._write_config(server_url, http_port, grpc_port)

        # Pull image if needed
        try:
            self.client.images.get(HEADSCALE_IMAGE)
        except NotFound:
            logger.info(f"Pulling {HEADSCALE_IMAGE}...")
            self.client.images.pull(HEADSCALE_IMAGE)

        # Create and start container
        logger.info("Creating Headscale container")
        try:
            self.client.containers.run(
                HEADSCALE_IMAGE,
                name=HEADSCALE_CONTAINER_NAME,
                command="serve",
                detach=True,
                volumes={
                    HEADSCALE_DATA_DIR: {"bind": "/etc/headscale", "mode": "rw"},
                    f"{HEADSCALE_DATA_DIR}/run": {"bind": "/var/run/headscale", "mode": "rw"},
                },
                ports={
                    f"{http_port}/tcp": ("0.0.0.0", http_port),
                    f"{grpc_port}/tcp": ("0.0.0.0", grpc_port),
                },
                restart_policy={"Name": "unless-stopped"},
            )

            # Wait for Headscale to start
            await asyncio.sleep(3)

            # Create default user
            await self._create_user(LMSTACK_USER)

            logger.info("Headscale started successfully")
            return True

        except APIError as e:
            logger.error(f"Failed to start Headscale: {e}")
            return False

    async def stop(self) -> bool:
        """Stop Headscale server."""
        container = self._get_container()
        if container:
            container.stop()
            logger.info("Headscale stopped")
            return True
        return False

    async def remove(self) -> bool:
        """Stop and remove Headscale container."""
        container = self._get_container()
        if container:
            container.stop()
            container.remove()
            logger.info("Headscale container removed")
            return True
        return False

    async def _exec_command(self, *args) -> tuple[int, str]:
        """Execute a headscale command in the container."""
        container = self._get_container()
        if not container or container.status != "running":
            raise RuntimeError("Headscale container is not running")

        cmd = ["headscale", *args]
        result = container.exec_run(cmd)
        return result.exit_code, result.output.decode("utf-8")

    async def _create_user(self, username: str) -> bool:
        """Create a user in Headscale."""
        exit_code, output = await self._exec_command("users", "create", username)
        if exit_code == 0 or "already exists" in output.lower():
            logger.info(f"User '{username}' created or already exists")
            return True
        logger.error(f"Failed to create user: {output}")
        return False

    async def _get_user_id(self, username: str) -> Optional[int]:
        """Get user ID by username."""
        exit_code, output = await self._exec_command("users", "list", "--output", "json")
        if exit_code == 0:
            try:
                users = json.loads(output)
                if users:
                    for user in users:
                        if user.get("name") == username:
                            return user.get("id")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse users JSON: {output}")
        return None

    async def create_preauth_key(
        self,
        user: str = LMSTACK_USER,
        reusable: bool = True,
        ephemeral: bool = False,
        expiration: str = "24h",
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Create a pre-authentication key for joining nodes."""
        # Get user ID (newer Headscale versions require user ID instead of name)
        user_id = await self._get_user_id(user)
        if user_id is None:
            logger.error(f"User '{user}' not found")
            return None

        args = ["preauthkeys", "create", "--user", str(user_id)]

        if reusable:
            args.append("--reusable")
        if ephemeral:
            args.append("--ephemeral")
        if expiration:
            args.extend(["--expiration", expiration])
        if tags:
            for tag in tags:
                args.extend(["--tags", f"tag:{tag}"])

        exit_code, output = await self._exec_command(*args)

        if exit_code == 0:
            # Parse the key from output
            # Output format varies, try to find the key
            lines = output.strip().split("\n")
            for line in lines:
                # Key is usually a long alphanumeric string
                if len(line) > 40 and " " not in line.strip():
                    return line.strip()
                # Or it might be in the last column of a table row
                parts = line.split()
                if parts:
                    potential_key = parts[-1]
                    if len(potential_key) > 40:
                        return potential_key

            # If we couldn't parse, log the output
            logger.warning(f"Could not parse preauth key from output: {output}")
            return output.strip()

        logger.error(f"Failed to create preauth key: {output}")
        return None

    async def list_nodes(self, user: str = LMSTACK_USER) -> List[HeadscaleNode]:
        """List all nodes (newer Headscale versions don't support --user filter)."""
        exit_code, output = await self._exec_command("nodes", "list", "--output", "json")

        if exit_code == 0:
            try:
                data = json.loads(output)
                if data is None:
                    return []
                return [HeadscaleNode(node) for node in data]
            except json.JSONDecodeError:
                logger.error(f"Failed to parse nodes JSON: {output}")
                return []

        logger.error(f"Failed to list nodes: {output}")
        return []

    async def get_node(self, node_id: int) -> Optional[HeadscaleNode]:
        """Get a specific node by ID."""
        nodes = await self.list_nodes()
        for node in nodes:
            if node.id == node_id:
                return node
        return None

    async def get_node_by_name(self, name: str) -> Optional[HeadscaleNode]:
        """Get a node by name."""
        nodes = await self.list_nodes()
        for node in nodes:
            if node.name == name or node.given_name == name:
                return node
        return None

    async def delete_node(self, node_id: int) -> bool:
        """Delete a node from Headscale."""
        exit_code, output = await self._exec_command("nodes", "delete", "--identifier", str(node_id), "--force")
        if exit_code == 0:
            logger.info(f"Node {node_id} deleted")
            return True
        logger.error(f"Failed to delete node: {output}")
        return False

    async def rename_node(self, node_id: int, new_name: str) -> bool:
        """Rename a node."""
        exit_code, output = await self._exec_command("nodes", "rename", "--identifier", str(node_id), new_name)
        if exit_code == 0:
            logger.info(f"Node {node_id} renamed to {new_name}")
            return True
        logger.error(f"Failed to rename node: {output}")
        return False

    def get_join_command(self, preauth_key: str, server_url: Optional[str] = None) -> str:
        """Get the command for a node to join the network."""
        url = server_url or f"{self.server_url}:{self.http_port}"
        return f"tailscale up --login-server {url} --authkey {preauth_key}"

    async def get_status(self) -> dict:
        """Get Headscale status."""
        container = self._get_container()

        status = {
            "enabled": container is not None,
            "running": container is not None and container.status == "running",
            "container_status": container.status if container else None,
            "server_url": f"{self.server_url}:{self.http_port}" if container else None,
        }

        if status["running"]:
            nodes = await self.list_nodes()
            status["nodes_count"] = len(nodes)
            status["online_nodes"] = len([n for n in nodes if n.online])

        return status


# Global instance
_headscale_manager: Optional[HeadscaleManager] = None


def get_headscale_manager() -> HeadscaleManager:
    """Get or create Headscale manager instance."""
    global _headscale_manager
    if _headscale_manager is None:
        _headscale_manager = HeadscaleManager()
    return _headscale_manager

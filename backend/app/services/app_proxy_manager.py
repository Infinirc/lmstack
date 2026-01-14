"""App Proxy Manager

Manages nginx reverse proxy for deployed apps.
Runs nginx container on the LMStack controller to proxy requests to worker apps.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import docker
from docker.errors import NotFound, APIError

logger = logging.getLogger(__name__)

NGINX_CONTAINER_NAME = "lmstack-app-proxy"
NGINX_IMAGE = "nginx:alpine"
PROXY_CONFIG_DIR = "/tmp/lmstack-proxy"
NGINX_CONF_PATH = f"{PROXY_CONFIG_DIR}/nginx.conf"
NGINX_CONFD_PATH = f"{PROXY_CONFIG_DIR}/conf.d"


def get_docker_client() -> docker.DockerClient:
    """Get Docker client."""
    return docker.from_env()


def generate_nginx_main_conf() -> str:
    """Generate main nginx.conf."""
    return """
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    keepalive_timeout 65;
    client_max_body_size 100M;

    include /etc/nginx/conf.d/*.conf;
}
"""


def generate_app_proxy_conf(
    app_id: int,
    app_type: str,
    listen_port: int,
    worker_host: str,
    worker_port: int,
) -> str:
    """Generate nginx config for a single app proxy."""
    return f"""# Proxy config for app {app_id} ({app_type})
server {{
    listen {listen_port};
    server_name _;

    location / {{
        proxy_pass http://{worker_host}:{worker_port};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        proxy_buffering off;
    }}
}}
"""


class AppProxyManager:
    """Manages nginx reverse proxy for deployed apps."""

    def __init__(self):
        self._client: Optional[docker.DockerClient] = None
        self._setup_config_dir()

    def _setup_config_dir(self):
        """Setup config directories."""
        os.makedirs(NGINX_CONFD_PATH, exist_ok=True)
        # Write main nginx.conf
        with open(NGINX_CONF_PATH, "w") as f:
            f.write(generate_nginx_main_conf())

    @property
    def client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            self._client = get_docker_client()
        return self._client

    def _get_container(self) -> Optional[docker.models.containers.Container]:
        """Get nginx container if exists."""
        try:
            return self.client.containers.get(NGINX_CONTAINER_NAME)
        except NotFound:
            return None

    def _get_used_ports(self) -> set[int]:
        """Get ports currently used by nginx proxy configs."""
        ports = set()
        confd = Path(NGINX_CONFD_PATH)
        if confd.exists():
            for conf_file in confd.glob("*.conf"):
                content = conf_file.read_text()
                # Extract listen port from config
                import re

                match = re.search(r"listen\s+(\d+)", content)
                if match:
                    ports.add(int(match.group(1)))
        return ports

    async def ensure_running(self) -> bool:
        """Ensure nginx container is running."""
        container = self._get_container()

        if container:
            if container.status != "running":
                logger.info("Starting existing nginx proxy container")
                container.start()
            return True

        # Pull image if needed
        try:
            self.client.images.get(NGINX_IMAGE)
        except NotFound:
            logger.info(f"Pulling {NGINX_IMAGE}...")
            self.client.images.pull(NGINX_IMAGE)

        # Get all ports from existing configs
        ports = self._get_used_ports()
        port_bindings = {f"{p}/tcp": ("0.0.0.0", p) for p in ports}

        # Start container
        logger.info("Creating nginx proxy container")
        try:
            self.client.containers.run(
                NGINX_IMAGE,
                name=NGINX_CONTAINER_NAME,
                detach=True,
                volumes={
                    NGINX_CONF_PATH: {"bind": "/etc/nginx/nginx.conf", "mode": "ro"},
                    NGINX_CONFD_PATH: {"bind": "/etc/nginx/conf.d", "mode": "ro"},
                },
                ports=port_bindings,
                restart_policy={"Name": "unless-stopped"},
            )
            return True
        except APIError as e:
            logger.error(f"Failed to create nginx container: {e}")
            return False

    async def add_app_proxy(
        self,
        app_id: int,
        app_type: str,
        listen_port: int,
        worker_host: str,
        worker_port: int,
    ) -> bool:
        """Add proxy config for an app and reload nginx."""
        # Generate config file
        config = generate_app_proxy_conf(
            app_id=app_id,
            app_type=app_type,
            listen_port=listen_port,
            worker_host=worker_host,
            worker_port=worker_port,
        )

        config_file = Path(NGINX_CONFD_PATH) / f"app_{app_id}.conf"
        config_file.write_text(config)
        logger.info(
            f"Created proxy config for app {app_id}: port {listen_port} -> {worker_host}:{worker_port}"
        )

        # Recreate container with new port binding
        container = self._get_container()
        if container:
            # Stop and remove old container
            container.stop()
            container.remove()

        # Start with all port bindings
        ports = self._get_used_ports()
        port_bindings = {f"{p}/tcp": ("0.0.0.0", p) for p in ports}

        try:
            self.client.containers.run(
                NGINX_IMAGE,
                name=NGINX_CONTAINER_NAME,
                detach=True,
                volumes={
                    NGINX_CONF_PATH: {"bind": "/etc/nginx/nginx.conf", "mode": "ro"},
                    NGINX_CONFD_PATH: {"bind": "/etc/nginx/conf.d", "mode": "ro"},
                },
                ports=port_bindings,
                restart_policy={"Name": "unless-stopped"},
            )
            logger.info(f"Nginx proxy container started with ports: {list(ports)}")
            return True
        except APIError as e:
            logger.error(f"Failed to start nginx container: {e}")
            return False

    async def remove_app_proxy(self, app_id: int) -> bool:
        """Remove proxy config for an app and reload nginx."""
        config_file = Path(NGINX_CONFD_PATH) / f"app_{app_id}.conf"
        if config_file.exists():
            config_file.unlink()
            logger.info(f"Removed proxy config for app {app_id}")

        # Recreate container with updated port bindings
        container = self._get_container()
        if container:
            container.stop()
            container.remove()

        ports = self._get_used_ports()
        if not ports:
            # No more apps, don't restart nginx
            logger.info("No more app proxies, nginx container removed")
            return True

        port_bindings = {f"{p}/tcp": ("0.0.0.0", p) for p in ports}
        try:
            self.client.containers.run(
                NGINX_IMAGE,
                name=NGINX_CONTAINER_NAME,
                detach=True,
                volumes={
                    NGINX_CONF_PATH: {"bind": "/etc/nginx/nginx.conf", "mode": "ro"},
                    NGINX_CONFD_PATH: {"bind": "/etc/nginx/conf.d", "mode": "ro"},
                },
                ports=port_bindings,
                restart_policy={"Name": "unless-stopped"},
            )
            return True
        except APIError as e:
            logger.error(f"Failed to restart nginx container: {e}")
            return False

    async def stop(self):
        """Stop and remove nginx proxy container."""
        container = self._get_container()
        if container:
            container.stop()
            container.remove()
            logger.info("Nginx proxy container stopped and removed")


# Global instance
_proxy_manager: Optional[AppProxyManager] = None


def get_proxy_manager() -> AppProxyManager:
    """Get or create proxy manager instance."""
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = AppProxyManager()
    return _proxy_manager

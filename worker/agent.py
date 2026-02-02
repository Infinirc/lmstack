"""LMStack Worker Agent - runs on GPU nodes to manage model deployments.

This is the main entry point for the worker agent. It handles:
- Registration with the LMStack server
- Heartbeat management
- FastAPI application setup with routes from the routes package
"""

import argparse
import asyncio
import logging
import os
import signal
import socket
import sys
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI

# Import from docker_ops package - support both direct run and module run
try:
    from docker_ops import ContainerManager, DockerRunner, GPUDetector, ImageManager, SystemDetector
    from routes import (
        containers_router,
        converter_router,
        deployment_router,
        images_router,
        native_router,
        storage_router,
    )
    from routes.containers import set_agent as set_containers_agent
    from routes.converter import set_agent as set_converter_agent
    from routes.deployment import set_agent as set_deployment_agent
    from routes.images import set_agent as set_images_agent
    from routes.native import set_agent as set_native_agent
    from routes.storage import set_agent as set_storage_agent
except ImportError:
    # Fallback for when running as package
    from worker.docker_ops import (
        ContainerManager,
        DockerRunner,
        GPUDetector,
        ImageManager,
        SystemDetector,
    )
    from worker.routes import (
        containers_router,
        converter_router,
        deployment_router,
        images_router,
        native_router,
        storage_router,
    )
    from worker.routes.containers import set_agent as set_containers_agent
    from worker.routes.converter import set_agent as set_converter_agent
    from worker.routes.deployment import set_agent as set_deployment_agent
    from worker.routes.images import set_agent as set_images_agent
    from worker.routes.native import set_agent as set_native_agent
    from worker.routes.storage import set_agent as set_storage_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WorkerAgent:
    """Worker agent that manages deployments on this node."""

    def __init__(
        self,
        name: str,
        server_url: str,
        host: str = "0.0.0.0",
        port: int = 52001,
        registration_token: Optional[str] = None,
    ):
        self.name = name
        self.server_url = server_url.rstrip("/")
        self.host = host
        self.port = port
        self.registration_token = registration_token
        self.worker_id: Optional[int] = None

        # Initialize detectors
        self.gpu_detector = GPUDetector()
        self.system_detector = SystemDetector()

        # Check capabilities
        self.capabilities = self.system_detector.detect_capabilities()
        self.os_type = self.capabilities.get("os_type", "linux")
        self.has_docker = self.capabilities.get("docker", True)

        # Initialize Docker managers (if Docker is available)
        if self.has_docker:
            self.docker = DockerRunner()
            self.image_manager = ImageManager(self.docker.client)
            self.container_manager = ContainerManager(self.docker.client)
        else:
            self.docker = None
            self.image_manager = None
            self.container_manager = None
            logger.info("Docker not available, using native process management")

        # Initialize native process manager for Mac (always, for Ollama support)
        self.native_manager = None
        if self.os_type == "darwin":
            try:
                from native_ops import NativeProcessManager
            except ImportError:
                from worker.native_ops import NativeProcessManager
            self.native_manager = NativeProcessManager()
            logger.info("Initialized native process manager for macOS")

        # Task management
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = True

    def _is_local_worker(self) -> bool:
        """Check if this worker is running on the same machine as the server."""
        # Check if server_url contains localhost or host.docker.internal
        local_indicators = ["localhost", "127.0.0.1", "host.docker.internal", "::1"]
        server_host = self.server_url.split("://")[-1].split(":")[0].split("/")[0].lower()
        return server_host in local_indicators

    async def register(self) -> bool:
        """Register this worker with the server."""
        try:
            # For Mac workers, ensure Ollama is running with external access
            if self.os_type == "darwin" and self.native_manager:
                if self.capabilities.get("ollama"):
                    logger.info("Ensuring Ollama service is running with external access...")
                    await self.native_manager.ensure_ollama_running()

            gpu_info = self.gpu_detector.detect()
            system_info = self.system_detector.detect()

            # Build labels - mark as local if connecting to localhost/host.docker.internal
            labels = {}
            if self._is_local_worker():
                labels["type"] = "local"
                logger.info("Detected local worker (same machine as LMStack server)")

            registration_data = {
                "name": self.name,
                "address": f"{self._get_advertise_address()}:{self.port}",
                "gpu_info": gpu_info,
                "system_info": system_info,
                "labels": labels if labels else None,
            }

            # Include registration token if provided
            if self.registration_token:
                registration_data["registration_token"] = self.registration_token

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.server_url}/api/workers",
                    json=registration_data,
                )

                if response.status_code == 201:
                    data = response.json()
                    self.worker_id = data["id"]
                    logger.info(f"Registered as worker {self.worker_id}")
                    return True
                elif response.status_code == 400:
                    # Worker might already exist, try to reconnect
                    logger.warning("Worker already exists, attempting to reconnect...")
                    return await self._reconnect()
                elif response.status_code == 401:
                    error_detail = response.json().get("detail", "Authentication failed")
                    logger.error(f"Registration failed: {error_detail}")
                    return False
                else:
                    logger.error(f"Failed to register: {response.text}")
                    return False

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
        except httpx.TimeoutException as e:
            logger.error(f"Registration timed out: {e}")
            return False

    async def _reconnect(self) -> bool:
        """Try to reconnect to existing worker record."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.server_url}/api/workers")
                if response.status_code == 200:
                    workers = response.json()["items"]
                    for worker in workers:
                        if worker["name"] == self.name:
                            self.worker_id = worker["id"]
                            # Update address
                            await client.patch(
                                f"{self.server_url}/api/workers/{self.worker_id}",
                                json={
                                    "address": f"{self._get_advertise_address()}:{self.port}",
                                    "status": "online",
                                },
                            )
                            logger.info(f"Reconnected as worker {self.worker_id}")
                            return True
            return False
        except httpx.HTTPError as e:
            logger.error(f"Failed to reconnect: {e}")
            return False

    def _get_advertise_address(self) -> str:
        """Get the address to advertise to the server."""
        # Try to get from environment
        addr = os.environ.get("ADVERTISE_ADDRESS")
        if addr:
            return addr

        # Try to get local IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            addr = s.getsockname()[0]
            s.close()
            return addr
        except OSError:
            return "127.0.0.1"

    async def heartbeat_loop(self):
        """Send periodic heartbeat to server."""
        while self._running:
            try:
                await asyncio.sleep(10)  # 10 seconds interval
                if not self._running:
                    break

                gpu_info = self.gpu_detector.detect()
                system_info = self.system_detector.detect()

                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"{self.server_url}/api/workers/heartbeat",
                        json={
                            "worker_id": self.worker_id,
                            "status": "online",
                            "gpu_info": gpu_info,
                            "system_info": system_info,
                        },
                    )
                    if response.status_code != 200:
                        logger.warning(f"Heartbeat failed: {response.text}")

            except asyncio.CancelledError:
                break
            except httpx.HTTPError as e:
                logger.warning(f"Heartbeat error: {e}")

    def _notify_offline_sync(self):
        """Synchronously notify server that this worker is going offline."""
        if not self.worker_id:
            return

        try:
            # Use synchronous httpx to ensure we wait for the response
            import httpx as httpx_sync

            with httpx_sync.Client(timeout=5.0) as client:
                response = client.post(
                    f"{self.server_url}/api/workers/heartbeat",
                    json={
                        "worker_id": self.worker_id,
                        "status": "offline",
                        "gpu_info": [],
                        "system_info": {},
                    },
                )
                if response.status_code == 200:
                    logger.info("Notified server of shutdown")
                else:
                    logger.warning(f"Failed to notify server: {response.text}")
        except Exception as e:
            logger.warning(f"Could not notify server of shutdown: {e}")

    def shutdown(self):
        """Shutdown the agent."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Notify server we're going offline synchronously
        self._notify_offline_sync()


# Global agent instance
agent: Optional[WorkerAgent] = None


def _set_agent_references(worker_agent: WorkerAgent):
    """Set agent references for all route modules."""
    set_deployment_agent(worker_agent)
    set_images_agent(worker_agent)
    set_containers_agent(worker_agent)
    set_storage_agent(worker_agent)
    set_native_agent(worker_agent)
    set_converter_agent(worker_agent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global agent

    if agent:
        # Set agent references for routes
        _set_agent_references(agent)

        # Recover native processes from previous run (Mac only)
        if agent.native_manager:
            logger.info("Attempting to recover native processes from previous run...")
            try:
                results = await agent.native_manager.recover_processes()
                if results:
                    recovered = sum(1 for v in results.values() if v == "recovered")
                    if recovered > 0:
                        logger.info(f"Recovered {recovered} native process(es)")
            except Exception as e:
                logger.warning(f"Failed to recover native processes: {e}")

        # Register with server
        registered = await agent.register()
        if not registered:
            logger.warning("Failed to register with server, continuing anyway...")

        # Start heartbeat loop
        agent._heartbeat_task = asyncio.create_task(agent.heartbeat_loop())

    yield

    # Shutdown
    if agent:
        agent.shutdown()


app = FastAPI(title="LMStack Worker Agent", lifespan=lifespan)

# Include routers
app.include_router(deployment_router)
app.include_router(images_router)
app.include_router(containers_router)
app.include_router(storage_router)
app.include_router(native_router)
app.include_router(converter_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "worker_id": agent.worker_id if agent else None}


def main():
    global agent

    # Get defaults from environment variables (for Docker)
    default_name = os.environ.get("WORKER_NAME")
    default_server_url = os.environ.get("BACKEND_URL")
    default_registration_token = os.environ.get("REGISTRATION_TOKEN")

    parser = argparse.ArgumentParser(description="LMStack Worker Agent")
    parser.add_argument(
        "--name",
        default=default_name,
        required=default_name is None,
        help="Worker name (or set WORKER_NAME env var)",
    )
    parser.add_argument(
        "--server-url",
        default=default_server_url,
        required=default_server_url is None,
        help="LMStack server URL (or set BACKEND_URL env var)",
    )
    parser.add_argument(
        "--registration-token",
        default=default_registration_token,
        help="Registration token from the server (or set REGISTRATION_TOKEN env var)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Agent host")
    default_port = int(os.environ.get("AGENT_PORT", "52001"))
    parser.add_argument("--port", type=int, default=default_port, help="Agent port")

    args = parser.parse_args()

    agent = WorkerAgent(
        name=args.name,
        server_url=args.server_url,
        host=args.host,
        port=args.port,
        registration_token=args.registration_token,
    )

    # Handle shutdown signals
    def handle_signal(signum, frame):
        logger.info("Received shutdown signal")
        agent.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Starting worker agent '{args.name}' on {args.host}:{args.port}")
    logger.info(f"Server URL: {args.server_url}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

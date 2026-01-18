"""Local Worker Service

Utilities for registering the local machine as a worker.
"""

import logging
import platform
import socket
import subprocess

import psutil

logger = logging.getLogger(__name__)


def get_local_hostname() -> str:
    """Get the local hostname."""
    return socket.gethostname()


def get_local_ip() -> str:
    """Get the local IP address."""
    try:
        # Connect to external server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError as e:
        logger.debug(f"Could not determine local IP: {e}")
        return "127.0.0.1"


def get_gpu_info() -> list[dict]:
    """Get GPU information using nvidia-smi."""
    gpus = []
    try:
        # Try nvidia-smi first
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    gpus.append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total": int(float(parts[2]) * 1024 * 1024),  # MB to bytes
                            "memory_used": int(float(parts[3]) * 1024 * 1024),
                            "memory_free": int(float(parts[4]) * 1024 * 1024),
                            "utilization": int(parts[5]) if parts[5] != "[N/A]" else 0,
                            "temperature": int(parts[6]) if parts[6] != "[N/A]" else 0,
                        }
                    )
    except FileNotFoundError:
        logger.info("nvidia-smi not found, no NVIDIA GPU detected")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")

    return gpus


def get_system_info() -> dict:
    """Get system information (CPU, memory, disk)."""
    # CPU info
    cpu_info = {
        "percent": psutil.cpu_percent(interval=0.1),
        "count": psutil.cpu_count(),
        "freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
    }

    # Memory info
    mem = psutil.virtual_memory()
    memory_info = {
        "total": mem.total,
        "used": mem.used,
        "free": mem.available,
        "percent": mem.percent,
    }

    # Disk info (root partition)
    disk = psutil.disk_usage("/")
    disk_info = {
        "total": disk.total,
        "used": disk.used,
        "free": disk.free,
        "percent": disk.percent,
    }

    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "disk": disk_info,
    }


def get_local_worker_info() -> dict:
    """Get all local worker information."""
    return {
        "hostname": get_local_hostname(),
        "ip": get_local_ip(),
        "gpu_info": get_gpu_info(),
        "system_info": get_system_info(),
        "platform": platform.system(),
        "platform_release": platform.release(),
    }


def spawn_docker_worker(
    worker_name: str,
    backend_url: str,
    registration_token: str,
    container_name: str = "lmstack-worker",
) -> dict:
    """Spawn a Docker worker container on the local machine.

    Returns:
        dict with keys: success, message, container_id (if success)
    """
    # First, check if container with same name exists and remove it
    try:
        check_result = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f", f"name=^{container_name}$"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if check_result.stdout.strip():
            # Container exists, stop and remove it
            logger.info(f"Removing existing container: {container_name}")
            subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["docker", "rm", container_name],
                capture_output=True,
                timeout=10,
            )
    except subprocess.TimeoutExpired:
        logger.warning("Timeout while checking/removing existing container")
    except Exception as e:
        logger.warning(f"Error checking existing container: {e}")

    # Build the docker run command
    # Use --network host so worker can access backend and deployed apps
    # Use --restart unless-stopped so worker auto-starts after reboot
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--restart",
        "unless-stopped",
        "--network",
        "host",
        "--gpus",
        "all",
        "--privileged",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
        "-v",
        f"{_get_huggingface_cache()}:/root/.cache/huggingface",
        "-v",
        "/:/host:ro",
        "-e",
        f"BACKEND_URL={backend_url}",
        "-e",
        f"WORKER_NAME={worker_name}",
        "-e",
        f"REGISTRATION_TOKEN={registration_token}",
        "infinirc/lmstack-worker:latest",
    ]

    try:
        logger.info(f"Spawning Docker worker: {worker_name}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            container_id = result.stdout.strip()[:12]
            logger.info(f"Docker worker spawned successfully: {container_id}")
            return {
                "success": True,
                "message": f"Worker container started: {container_id}",
                "container_id": container_id,
            }
        else:
            error_msg = result.stderr.strip() or result.stdout.strip()
            logger.error(f"Failed to spawn Docker worker: {error_msg}")
            return {
                "success": False,
                "message": f"Failed to start worker: {error_msg}",
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "message": "Timeout while starting Docker container",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "message": "Docker not found. Please install Docker first.",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error starting Docker worker: {str(e)}",
        }


def _get_huggingface_cache() -> str:
    """Get the HuggingFace cache directory path."""
    import os

    # Check HF_HOME first, then default to ~/.cache/huggingface
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return hf_home
    return os.path.expanduser("~/.cache/huggingface")


def stop_docker_worker(container_name: str = "lmstack-worker") -> dict:
    """Stop and remove a Docker worker container.

    Returns:
        dict with keys: success, message
    """
    try:
        # Check if container exists
        check_result = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f", f"name=^{container_name}$"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if not check_result.stdout.strip():
            return {
                "success": True,
                "message": f"Container {container_name} not found (already removed)",
            }

        # Stop the container
        logger.info(f"Stopping container: {container_name}")
        stop_result = subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Remove the container
        logger.info(f"Removing container: {container_name}")
        rm_result = subprocess.run(
            ["docker", "rm", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if rm_result.returncode == 0:
            logger.info(f"Container {container_name} stopped and removed")
            return {
                "success": True,
                "message": f"Container {container_name} stopped and removed",
            }
        else:
            error_msg = rm_result.stderr.strip() or stop_result.stderr.strip()
            return {
                "success": False,
                "message": f"Failed to remove container: {error_msg}",
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "message": "Timeout while stopping Docker container",
        }
    except FileNotFoundError:
        return {
            "success": True,
            "message": "Docker not found (container may not exist)",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error stopping Docker worker: {str(e)}",
        }

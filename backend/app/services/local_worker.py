"""Local Worker Service

Utilities for registering the local machine as a worker.
"""

import logging
import platform
import socket
import subprocess
from typing import Optional

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
    except (OSError, socket.error) as e:
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

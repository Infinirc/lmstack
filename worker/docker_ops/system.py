"""System information detection for LMStack Worker.

Provides CPU, RAM, and disk usage information.
Supports both host and container environments.
Also detects OS type and available native backends for Mac support.
"""

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class SystemDetector:
    """Detect system information (CPU, RAM, Disk).

    Supports container-aware detection by reading from cgroups
    and using mounted host paths when available.
    """

    def __init__(self, disk_path: str = "/"):
        """Initialize detector.

        Args:
            disk_path: Path to check disk usage. Use /host for mounted host root.
        """
        # Check if /host is mounted (common pattern for host filesystem access)
        if os.path.exists("/host") and os.path.ismount("/host"):
            self.disk_path = "/host"
        else:
            self.disk_path = disk_path

    def _read_cgroup_file(self, *paths: str) -> str | None:
        """Try to read from cgroup files (v1 and v2 compatible)."""
        for path in paths:
            try:
                if os.path.exists(path):
                    return Path(path).read_text().strip()
            except (OSError, IOError):
                continue
        return None

    def _get_memory_info(self) -> dict:
        """Get memory info, preferring cgroup limits in containers."""
        import psutil

        # Try cgroups v2 first
        mem_max = self._read_cgroup_file("/sys/fs/cgroup/memory.max")
        mem_current = self._read_cgroup_file("/sys/fs/cgroup/memory.current")

        # Try cgroups v1
        if not mem_max:
            mem_max = self._read_cgroup_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            mem_current = self._read_cgroup_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")

        # Get host memory for reference
        host_mem = psutil.virtual_memory()

        # If cgroup limit is set and not "max" (unlimited)
        if mem_max and mem_max != "max" and mem_max != "9223372036854771712":
            try:
                total = int(mem_max)
                used = int(mem_current) if mem_current else 0
                # If cgroup limit is reasonable (less than host memory), use it
                if total < host_mem.total:
                    return {
                        "total": total,
                        "used": used,
                        "free": total - used,
                        "percent": round(used / total * 100, 1) if total > 0 else 0,
                    }
            except (ValueError, ZeroDivisionError):
                pass

        # Fall back to psutil (shows host memory)
        return {
            "total": host_mem.total,
            "used": host_mem.used,
            "free": host_mem.available,
            "percent": host_mem.percent,
        }

    def _get_cpu_info(self) -> dict:
        """Get CPU info."""
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # Try to get cgroup CPU quota (for containers with CPU limits)
        quota = self._read_cgroup_file("/sys/fs/cgroup/cpu.max")
        if quota and quota != "max":
            try:
                parts = quota.split()
                if len(parts) == 2 and parts[0] != "max":
                    quota_us = int(parts[0])
                    period_us = int(parts[1])
                    effective_cpus = quota_us / period_us
                    if effective_cpus < cpu_count:
                        cpu_count = round(effective_cpus, 1)
            except (ValueError, IndexError):
                pass

        return {
            "percent": cpu_percent,
            "count": cpu_count,
            "freq_mhz": cpu_freq.current if cpu_freq else 0,
        }

    def _get_disk_info(self) -> dict:
        """Get disk info from configured path."""
        import psutil

        try:
            disk = psutil.disk_usage(self.disk_path)
            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            }
        except OSError:
            # Fall back to root if configured path fails
            disk = psutil.disk_usage("/")
            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            }

    def _get_os_type(self) -> str:
        """Detect operating system type."""
        system = platform.system().lower()
        if system == "darwin":
            return "darwin"
        elif system == "windows":
            return "windows"
        return "linux"

    def _get_gpu_type(self) -> str:
        """Detect GPU type."""
        os_type = self._get_os_type()

        if os_type == "darwin":
            # Check for Apple Silicon
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if "Apple" in result.stdout:
                    return "apple_silicon"
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
            # Fallback: check machine type
            if platform.machine() == "arm64":
                return "apple_silicon"
            return "none"

        # Check for NVIDIA GPU on Linux/Windows
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return "nvidia"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return "none"

    def _check_docker_available(self) -> bool:
        """Check if Docker is available and running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is installed."""
        return shutil.which("ollama") is not None

    def _check_ollama_running(self) -> bool:
        """Check if Ollama service is running."""
        try:
            import httpx

            response = httpx.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _check_mlx_available(self) -> bool:
        """Check if MLX-LM is installed (Mac only)."""
        if self._get_os_type() != "darwin":
            return False
        try:
            result = subprocess.run(
                ["python3", "-c", "import mlx_lm; print('ok')"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _check_llama_cpp_available(self) -> bool:
        """Check if llama.cpp server is available."""
        # Check for llama-server binary
        if shutil.which("llama-server"):
            return True
        # Check for llama-cpp-python
        try:
            result = subprocess.run(
                ["python3", "-c", "import llama_cpp; print('ok')"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def detect_capabilities(self) -> dict:
        """Detect available backends and capabilities.

        Returns:
            Dictionary with:
            - os_type: Operating system (linux, darwin, windows)
            - gpu_type: GPU type (nvidia, apple_silicon, amd, none)
            - docker: Whether Docker is available
            - ollama: Whether Ollama is installed
            - ollama_running: Whether Ollama service is running
            - mlx: Whether MLX-LM is available (Mac only)
            - llama_cpp: Whether llama.cpp is available
        """
        os_type = self._get_os_type()
        gpu_type = self._get_gpu_type()

        caps = {
            "os_type": os_type,
            "gpu_type": gpu_type,
            "docker": self._check_docker_available(),
            "ollama": self._check_ollama_available(),
            "ollama_running": self._check_ollama_running(),
        }

        # Mac-specific backends
        if os_type == "darwin":
            caps["mlx"] = self._check_mlx_available()
            caps["llama_cpp"] = self._check_llama_cpp_available()

        return caps

    def detect(self) -> dict:
        """Detect system CPU, RAM and disk usage.

        Returns:
            Dictionary with:
            - cpu: CPU info (percent, count, freq_mhz)
            - memory: RAM info (total, used, free, percent)
            - disk: Disk info (total, used, free, percent)
            - os_type: Operating system type
            - gpu_type: GPU type
            - capabilities: Available backends
        """
        try:
            import psutil  # noqa: F401 - ensure psutil is available

            caps = self.detect_capabilities()

            return {
                "cpu": self._get_cpu_info(),
                "memory": self._get_memory_info(),
                "disk": self._get_disk_info(),
                "os_type": caps.get("os_type", "linux"),
                "gpu_type": caps.get("gpu_type", "none"),
                "capabilities": caps,
            }

        except ImportError:
            logger.warning("psutil not installed, system info unavailable")
            return {}
        except Exception as e:
            logger.warning(f"System detection failed: {e}")
            return {}

"""System information detection for LMStack Worker.

Provides CPU, RAM, and disk usage information.
Supports both host and container environments.
"""

import logging
import os
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

    def detect(self) -> dict:
        """Detect system CPU, RAM and disk usage.

        Returns:
            Dictionary with:
            - cpu: CPU info (percent, count, freq_mhz)
            - memory: RAM info (total, used, free, percent)
            - disk: Disk info (total, used, free, percent)
        """
        try:
            import psutil  # noqa: F401 - ensure psutil is available

            return {
                "cpu": self._get_cpu_info(),
                "memory": self._get_memory_info(),
                "disk": self._get_disk_info(),
            }

        except ImportError:
            logger.warning("psutil not installed, system info unavailable")
            return {}
        except Exception as e:
            logger.warning(f"System detection failed: {e}")
            return {}

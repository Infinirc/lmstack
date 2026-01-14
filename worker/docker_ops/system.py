"""System information detection for LMStack Worker.

Provides CPU, RAM, and disk usage information using psutil.
"""

import logging

logger = logging.getLogger(__name__)


class SystemDetector:
    """Detect system information (CPU, RAM, Disk)."""

    def detect(self) -> dict:
        """Detect system CPU, RAM and disk usage.

        Returns:
            Dictionary with:
            - cpu: CPU info (percent, count, freq_mhz)
            - memory: RAM info (total, used, free, percent)
            - disk: Disk info (total, used, free, percent)
        """
        try:
            import psutil

            # CPU info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            # Memory info
            mem = psutil.virtual_memory()

            # Disk info (root partition)
            disk = psutil.disk_usage("/")

            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "freq_mhz": cpu_freq.current if cpu_freq else 0,
                },
                "memory": {
                    "total": mem.total,
                    "used": mem.used,
                    "free": mem.available,
                    "percent": mem.percent,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent,
                },
            }

        except ImportError:
            logger.warning("psutil not installed, system info unavailable")
            return {}
        except Exception as e:
            logger.warning(f"System detection failed: {e}")
            return {}

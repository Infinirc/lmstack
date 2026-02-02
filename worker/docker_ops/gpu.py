"""GPU detection for LMStack Worker.

Provides GPU detection for NVIDIA (using pynvml) and Apple Silicon.
"""

import json
import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


class GPUDetector:
    """Detect and report GPU information.

    Supports NVIDIA GPUs (via pynvml) and Apple Silicon (via system_profiler).
    """

    def detect(self) -> list[dict]:
        """Detect available GPUs.

        Returns:
            List of GPU information dictionaries with:
            - index: GPU index
            - name: GPU model name
            - memory_total: Total VRAM in bytes
            - memory_used: Used VRAM in bytes
            - memory_free: Free VRAM in bytes
            - utilization: GPU utilization percentage
            - temperature: GPU temperature in Celsius
        """
        # Check platform
        if platform.system() == "Darwin":
            return self._detect_apple_silicon()
        else:
            return self._detect_nvidia()

    def _detect_apple_silicon(self) -> list[dict]:
        """Detect Apple Silicon GPU information."""
        try:
            # Check if this is Apple Silicon
            machine = platform.machine()
            if machine != "arm64":
                # Intel Mac - no GPU info to report
                return []

            # Get GPU info from system_profiler
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning("system_profiler failed")
                return []

            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            if not displays:
                return []

            gpus = []
            for idx, display in enumerate(displays):
                gpu_name = display.get("sppci_model", "Apple Silicon GPU")

                # Get unified memory info (Apple Silicon uses unified memory)
                # Use psutil to get system memory as reference
                try:
                    import psutil

                    mem = psutil.virtual_memory()
                    # Apple Silicon GPU can use up to ~75% of unified memory for GPU tasks
                    # Report system memory info as reference
                    memory_total = mem.total
                    memory_used = mem.used
                    memory_free = mem.available
                except ImportError:
                    # Fallback: get memory from sysctl
                    try:
                        mem_result = subprocess.run(
                            ["sysctl", "-n", "hw.memsize"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        memory_total = int(mem_result.stdout.strip())
                        memory_used = 0
                        memory_free = memory_total
                    except Exception:
                        memory_total = 0
                        memory_used = 0
                        memory_free = 0

                gpus.append(
                    {
                        "index": idx,
                        "name": gpu_name,
                        "memory_total": memory_total,
                        "memory_used": memory_used,
                        "memory_free": memory_free,
                        "utilization": 0,  # Would need powermetrics (requires sudo)
                        "temperature": 0,  # Would need powermetrics (requires sudo)
                    }
                )

            return gpus

        except Exception as e:
            logger.warning(f"Apple Silicon GPU detection failed: {e}")
            return []

    def _detect_nvidia(self) -> list[dict]:
        """Detect NVIDIA GPUs using pynvml."""
        try:
            import pynvml

            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get GPU name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Get utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except pynvml.NVMLError:
                    gpu_util = 0

                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError:
                    temp = 0

                gpus.append(
                    {
                        "index": i,
                        "name": name,
                        "memory_total": mem_info.total,
                        "memory_used": mem_info.used,
                        "memory_free": mem_info.free,
                        "utilization": gpu_util,
                        "temperature": temp,
                    }
                )

            pynvml.nvmlShutdown()
            return gpus

        except ImportError:
            logger.debug("pynvml (nvidia-ml-py) not installed")
            return []
        except Exception as e:
            logger.warning(f"NVIDIA GPU detection failed: {e}")
            return []

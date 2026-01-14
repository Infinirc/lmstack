"""GPU detection for LMStack Worker.

Provides NVIDIA GPU detection using pynvml (nvidia-ml-py).
"""

import logging

logger = logging.getLogger(__name__)


class GPUDetector:
    """Detect and report GPU information using pynvml (nvidia-ml-py)."""

    def detect(self) -> list[dict]:
        """Detect available GPUs with temperature using pynvml.

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
                    name = name.decode('utf-8')

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
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except pynvml.NVMLError:
                    temp = 0

                gpus.append({
                    "index": i,
                    "name": name,
                    "memory_total": mem_info.total,
                    "memory_used": mem_info.used,
                    "memory_free": mem_info.free,
                    "utilization": gpu_util,
                    "temperature": temp,
                })

            pynvml.nvmlShutdown()
            return gpus

        except ImportError:
            logger.error("pynvml (nvidia-ml-py) not installed")
            return []
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return []

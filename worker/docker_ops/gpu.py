"""GPU detection for LMStack Worker.

Provides GPU detection for NVIDIA (using pynvml), Tegra/Jetson, and Apple Silicon.
"""

import json
import logging
import os
import platform
import subprocess

logger = logging.getLogger(__name__)


class GPUDetector:
    """Detect and report GPU information.

    Supports NVIDIA GPUs (via pynvml), Tegra/Jetson (via nvidia-smi), and Apple Silicon.
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
            # Try pynvml first (works on desktop NVIDIA GPUs)
            gpus = self._detect_nvidia_pynvml()
            if gpus:
                return gpus

            # Fallback to nvidia-smi (works on Tegra/Jetson/DGX Spark)
            return self._detect_nvidia_smi()

    def _is_tegra_platform(self) -> bool:
        """Check if this is a Tegra/Jetson platform."""
        # Check for Tegra release file
        if os.path.exists("/etc/nv_tegra_release"):
            return True

        # Check for Jetson model file
        if os.path.exists("/proc/device-tree/model"):
            try:
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().lower()
                    if "jetson" in model or "tegra" in model or "orin" in model:
                        return True
            except Exception:
                pass

        # Check for aarch64 + NVIDIA (DGX Spark, Jetson)
        if platform.machine() == "aarch64":
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return True
            except Exception:
                pass

        return False

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

    def _detect_nvidia_pynvml(self) -> list[dict]:
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
            logger.debug(f"pynvml GPU detection failed: {e}")
            return []

    def _detect_nvidia_smi(self) -> list[dict]:
        """Detect NVIDIA GPUs using nvidia-smi command.

        This is a fallback for Tegra/Jetson/DGX Spark where pynvml may not work.
        """
        try:
            # Query GPU info using nvidia-smi
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

            if result.returncode != 0:
                logger.debug(f"nvidia-smi failed: {result.stderr}")
                return []

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue

                try:
                    idx = int(parts[0])
                    name = parts[1]
                    # nvidia-smi returns memory in MiB, convert to bytes
                    memory_total = int(float(parts[2])) * 1024 * 1024
                    memory_used = int(float(parts[3])) * 1024 * 1024
                    memory_free = int(float(parts[4])) * 1024 * 1024
                    # Utilization may be [N/A] on some platforms
                    try:
                        utilization = int(float(parts[5]))
                    except (ValueError, TypeError):
                        utilization = 0
                    # Temperature may be [N/A] on some platforms
                    try:
                        temperature = int(float(parts[6]))
                    except (ValueError, TypeError):
                        temperature = 0

                    gpus.append(
                        {
                            "index": idx,
                            "name": name,
                            "memory_total": memory_total,
                            "memory_used": memory_used,
                            "memory_free": memory_free,
                            "utilization": utilization,
                            "temperature": temperature,
                        }
                    )
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse nvidia-smi output line: {line}, error: {e}")
                    continue

            if gpus:
                logger.info(f"Detected {len(gpus)} GPU(s) via nvidia-smi")
            return gpus

        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
            return []
        except Exception as e:
            logger.warning(f"nvidia-smi GPU detection failed: {e}")
            return []

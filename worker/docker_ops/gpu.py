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

    def _parse_nvidia_smi_value(self, value: str) -> float | None:
        """Parse a value from nvidia-smi output, returning None for [N/A] or invalid."""
        value = value.strip()
        if not value or value.startswith("[N/A]") or value == "N/A" or value == "Not Supported":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _get_unified_memory_info(self) -> tuple[int, int, int]:
        """Get system memory info for unified memory platforms (Tegra/DGX Spark).

        Returns (total, used, free) in bytes.
        """
        try:
            import psutil

            mem = psutil.virtual_memory()
            return mem.total, mem.used, mem.available
        except ImportError:
            pass

        # Fallback: read from /proc/meminfo
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        meminfo[key] = int(val) * 1024  # Convert kB to bytes

                total = meminfo.get("MemTotal", 0)
                free = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
                used = total - free
                return total, used, free
        except Exception:
            return 0, 0, 0

    def _get_gpu_process_memory(self, gpu_index: int = 0) -> int:
        """Get total GPU memory used by processes from nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_bus_id,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return 0

            total_mem = 0
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    val = self._parse_nvidia_smi_value(parts[-1])
                    if val is not None:
                        total_mem += int(val) * 1024 * 1024  # MiB to bytes
            return total_mem
        except Exception:
            return 0

    def _detect_nvidia_smi(self) -> list[dict]:
        """Detect NVIDIA GPUs using nvidia-smi command.

        This is a fallback for Tegra/Jetson/DGX Spark where pynvml may not work.
        Handles [N/A] memory values by using unified system memory info.
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
                except (ValueError, TypeError):
                    continue

                name = parts[1].strip()
                if not name:
                    continue

                # Parse memory values (may be [N/A] on unified memory platforms)
                mem_total = self._parse_nvidia_smi_value(parts[2])
                mem_used = self._parse_nvidia_smi_value(parts[3])
                mem_free = self._parse_nvidia_smi_value(parts[4])

                if mem_total is not None:
                    # nvidia-smi returns memory in MiB, convert to bytes
                    memory_total = int(mem_total) * 1024 * 1024
                    memory_used = int(mem_used or 0) * 1024 * 1024
                    memory_free = int(mem_free or 0) * 1024 * 1024
                else:
                    # Unified memory platform (Tegra/DGX Spark) - use system memory
                    logger.debug(
                        f"GPU {idx} ({name}) reports [N/A] memory, using unified system memory"
                    )
                    memory_total, memory_used, memory_free = self._get_unified_memory_info()
                    # Try to get actual GPU process memory usage
                    gpu_proc_mem = self._get_gpu_process_memory(idx)
                    if gpu_proc_mem > 0:
                        memory_used = gpu_proc_mem
                        memory_free = max(0, memory_total - memory_used)

                utilization_val = self._parse_nvidia_smi_value(parts[5])
                utilization = int(utilization_val) if utilization_val is not None else 0

                temperature_val = self._parse_nvidia_smi_value(parts[6])
                temperature = int(temperature_val) if temperature_val is not None else 0

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

            if gpus:
                logger.debug(f"Detected {len(gpus)} GPU(s) via nvidia-smi")
            return gpus

        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
            return []
        except Exception as e:
            logger.warning(f"nvidia-smi GPU detection failed: {e}")
            return []

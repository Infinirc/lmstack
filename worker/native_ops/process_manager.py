"""Native process manager for Mac workers.

Manages LLM inference processes without Docker for macOS with Apple Silicon.
Supports Ollama, MLX-LM, llama.cpp, and vLLM-Metal backends.
"""

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .converter import ModelConverter

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_PORT = 11434


@dataclass
class NativeProcess:
    """Represents a native process deployment."""

    process_id: str  # Unique identifier (deployment_id based)
    pid: int  # OS process ID
    backend: str  # ollama, mlx, llama_cpp, vllm
    model_id: str
    port: int
    process: Optional[subprocess.Popen] = None
    log_file: Optional[Path] = None  # Path to log file for this process


class NativeProcessManager:
    """Manages native processes for Mac LLM deployments."""

    def __init__(self):
        self._processes: dict[str, NativeProcess] = {}
        self._ollama_process: Optional[subprocess.Popen] = None
        self._converter = ModelConverter()
        self._log_dir = Path.home() / ".lmstack" / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    async def ensure_ollama_running(
        self, host: str = "0.0.0.0", port: int = OLLAMA_DEFAULT_PORT
    ) -> bool:
        """Ensure Ollama service is running and accessible.

        If Ollama is not running, starts it with OLLAMA_HOST set to allow external connections.

        Args:
            host: Host to bind to (default 0.0.0.0 for external access)
            port: Port to bind to (default 11434)

        Returns:
            True if Ollama is running and accessible
        """
        import httpx

        # Check if Ollama is already running
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"http://localhost:{port}/api/tags")
                if response.status_code == 200:
                    logger.info("Ollama service is already running")
                    return True
        except Exception:
            pass

        # Ollama not running, try to start it
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            logger.warning("Ollama is not installed")
            return False

        logger.info(f"Starting Ollama service on {host}:{port}")

        # Set environment for Ollama to bind to all interfaces
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{host}:{port}"

        try:
            # Start ollama serve in background
            self._ollama_process = subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            logger.info(f"Started Ollama service (PID {self._ollama_process.pid})")

            # Wait for Ollama to be ready
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        response = await client.get(f"http://localhost:{port}/api/tags")
                        if response.status_code == 200:
                            logger.info("Ollama service is ready")
                            return True
                except Exception:
                    pass

            logger.error("Ollama service failed to start in time")
            return False

        except Exception as e:
            logger.error(f"Failed to start Ollama service: {e}")
            return False

    def get_process(self, process_id: str) -> Optional[NativeProcess]:
        """Get a managed process by ID."""
        return self._processes.get(process_id)

    def list_processes(self) -> list[NativeProcess]:
        """List all managed processes."""
        return list(self._processes.values())

    async def start_process(
        self,
        process_id: str,
        backend: str,
        model_id: str,
        port: int,
        **kwargs,
    ) -> NativeProcess:
        """Start a native LLM process.

        Args:
            process_id: Unique identifier for this process
            backend: Backend type (ollama, mlx, llama_cpp)
            model_id: Model identifier (HF model ID or Ollama model name)
            port: Port to serve on
            **kwargs: Additional backend-specific arguments

        Returns:
            NativeProcess instance
        """
        if process_id in self._processes:
            # Return existing process if it's still valid
            existing = self._processes[process_id]
            if existing.model_id == model_id and existing.backend == backend:
                logger.info(f"Reusing existing process {process_id}")
                return existing
            # Remove old process if model/backend changed
            await self.stop_process(process_id)

        if backend == "ollama":
            process = await self._start_ollama(process_id, model_id, port, **kwargs)
        elif backend == "mlx":
            process = await self._start_mlx(process_id, model_id, port, **kwargs)
        elif backend == "llama_cpp":
            process = await self._start_llama_cpp(process_id, model_id, port, **kwargs)
        elif backend == "vllm":
            process = await self._start_vllm_metal(process_id, model_id, port, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._processes[process_id] = process
        logger.info(f"Started {backend} process {process_id} on port {port}")
        return process

    async def stop_process(self, process_id: str) -> bool:
        """Stop a managed process.

        Args:
            process_id: Process identifier

        Returns:
            True if stopped successfully
        """
        process = self._processes.get(process_id)
        if not process:
            logger.warning(f"Process {process_id} not found")
            return False

        try:
            if process.backend == "ollama":
                # For Ollama, we don't stop the service, just unload the model
                await self._unload_ollama_model(process)
            else:
                # For MLX and llama.cpp, terminate the process
                if process.process and process.process.poll() is None:
                    process.process.terminate()
                    try:
                        process.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.process.kill()

            del self._processes[process_id]
            logger.info(f"Stopped process {process_id}")
            return True

        except Exception as e:
            logger.error(f"Error stopping process {process_id}: {e}")
            return False

    def is_running(self, process_id: str) -> bool:
        """Check if a process is still running."""
        process = self._processes.get(process_id)
        if not process:
            return False

        if process.backend == "ollama":
            # Ollama service should always be running
            return True

        if process.process:
            return process.process.poll() is None

        return False

    async def _start_ollama(
        self,
        process_id: str,
        model_id: str,
        port: int,
        **kwargs,
    ) -> NativeProcess:
        """Start Ollama model serving.

        Ollama runs as a system service on port 11434.
        We pull the model and keep it loaded for serving.
        """
        import httpx

        ollama_port = OLLAMA_DEFAULT_PORT

        # Ensure Ollama service is running (starts it if needed)
        if not await self.ensure_ollama_running():
            raise RuntimeError(
                "Ollama service is not running and could not be started. "
                "Please install Ollama: https://ollama.ai"
            )

        # Pull the model if needed
        logger.info(f"Pulling Ollama model: {model_id}")
        async with httpx.AsyncClient(timeout=1800.0) as client:
            async with client.stream(
                "POST",
                f"http://localhost:{ollama_port}/api/pull",
                json={"name": model_id, "stream": True},
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            import json

                            data = json.loads(line)
                            status = data.get("status", "")
                            if status:
                                logger.debug(f"Ollama pull: {status}")
                        except Exception:
                            pass

        # Pre-load the model by sending a dummy request
        logger.info(f"Loading model into memory: {model_id}")
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                await client.post(
                    f"http://localhost:{ollama_port}/api/generate",
                    json={"model": model_id, "prompt": "hi", "stream": False},
                )
        except Exception as e:
            logger.warning(f"Model preload warning: {e}")

        return NativeProcess(
            process_id=process_id,
            pid=0,  # We don't manage Ollama's PID
            backend="ollama",
            model_id=model_id,
            port=ollama_port,
        )

    async def _unload_ollama_model(self, process: NativeProcess):
        """Unload an Ollama model from memory."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Send keep_alive: 0 to unload immediately
                await client.post(
                    f"http://localhost:{process.port}/api/generate",
                    json={
                        "model": process.model_id,
                        "prompt": "",
                        "keep_alive": 0,
                    },
                )
                logger.info(f"Unloaded Ollama model: {process.model_id}")
        except Exception as e:
            logger.warning(f"Failed to unload Ollama model: {e}")

    async def _ensure_mlx_lm_installed(self) -> str:
        """Ensure MLX-LM is installed in a virtual environment.

        Creates a virtual environment at ~/.lmstack/venvs/mlx-lm
        and installs mlx-lm if not already present.

        Returns:
            Path to the python command in the virtual environment
        """
        venv_dir = Path.home() / ".lmstack" / "venvs" / "mlx-lm"
        python_cmd = venv_dir / "bin" / "python"

        # Check if mlx-lm is already installed in venv
        if python_cmd.exists():
            # Verify mlx_lm is importable
            check = await asyncio.create_subprocess_exec(
                str(python_cmd),
                "-c",
                "import mlx_lm",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await check.wait()
            if check.returncode == 0:
                logger.info(f"MLX-LM already installed at {venv_dir}")
                return str(python_cmd)

        # Create virtual environment
        logger.info(f"Creating virtual environment for MLX-LM at {venv_dir}")
        venv_dir.parent.mkdir(parents=True, exist_ok=True)

        # Create venv
        create_venv = await asyncio.create_subprocess_exec(
            "python3",
            "-m",
            "venv",
            str(venv_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await create_venv.wait()

        if create_venv.returncode != 0:
            stderr = await create_venv.stderr.read()
            raise RuntimeError(f"Failed to create virtual environment: {stderr.decode()}")

        # Install mlx-lm
        pip_cmd = venv_dir / "bin" / "pip"
        logger.info("Installing mlx-lm (this may take a few minutes)...")

        install_proc = await asyncio.create_subprocess_exec(
            str(pip_cmd),
            "install",
            "mlx-lm",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await install_proc.communicate()

        if install_proc.returncode != 0:
            raise RuntimeError(f"Failed to install mlx-lm: {stderr.decode()}")

        logger.info("MLX-LM installed successfully")
        return str(python_cmd)

    async def _start_mlx(
        self,
        process_id: str,
        model_id: str,
        port: int,
        **kwargs,
    ) -> NativeProcess:
        """Start MLX-LM server for Apple Silicon.

        MLX-LM provides OpenAI-compatible API via mlx_lm.server.
        Automatically installs mlx-lm and converts models if needed.
        """
        # Ensure MLX-LM is installed (auto-install if needed)
        python_cmd = await self._ensure_mlx_lm_installed()

        effective_model_id = model_id

        # Check if model needs conversion
        if not ModelConverter.is_mlx_ready(model_id):
            # Check for cached conversion first
            cached = self._converter.get_cached_model(model_id, "mlx")
            if cached:
                logger.info(f"Using cached MLX model: {cached}")
                effective_model_id = cached
            else:
                # Try to find an existing MLX variant on HuggingFace
                mlx_variant = ModelConverter.find_mlx_variant(model_id)
                if mlx_variant and ModelConverter.is_mlx_ready(mlx_variant):
                    logger.info(f"Using MLX variant: {mlx_variant}")
                    effective_model_id = mlx_variant
                else:
                    # Convert the model
                    logger.info(f"Converting {model_id} to MLX format...")
                    try:
                        quantize = kwargs.pop("mlx_quantize", True)
                        bits = kwargs.pop("mlx_bits", 4)
                        effective_model_id = await self._converter.convert_to_mlx(
                            model_id, quantize=quantize, bits=bits
                        )
                    except Exception as e:
                        logger.error(f"MLX conversion failed: {e}")
                        raise RuntimeError(
                            f"Failed to convert model to MLX format: {e}. "
                            "Consider using an mlx-community model or Ollama backend."
                        )

        # Build command using venv python
        cmd = [
            python_cmd,
            "-m",
            "mlx_lm.server",
            "--model",
            effective_model_id,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]

        # Add optional parameters
        if kwargs.get("trust_remote_code"):
            cmd.append("--trust-remote-code")

        # Create log file
        log_file = self._log_dir / f"{process_id}.log"

        # Start the process with log file
        env = os.environ.copy()
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

        logger.info(f"Started MLX-LM server (PID {process.pid}) for {effective_model_id}")

        return NativeProcess(
            process_id=process_id,
            pid=process.pid,
            backend="mlx",
            model_id=effective_model_id,
            port=port,
            process=process,
            log_file=log_file,
        )

    async def _ensure_llama_cpp_installed(self) -> str:
        """Ensure llama.cpp is installed.

        Installs llama.cpp via Homebrew if not already present.

        Returns:
            Path to the llama-server command
        """
        llama_server = shutil.which("llama-server")
        if llama_server:
            logger.info(f"llama.cpp already installed at {llama_server}")
            return llama_server

        # Check if brew is available
        brew = shutil.which("brew")
        if not brew:
            raise RuntimeError(
                "llama-server not found and Homebrew is not installed. "
                "Please install Homebrew first: https://brew.sh"
            )

        # Install llama.cpp via brew
        logger.info("Installing llama.cpp via Homebrew (this may take a few minutes)...")

        install_proc = await asyncio.create_subprocess_exec(
            brew,
            "install",
            "llama.cpp",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await install_proc.communicate()

        if install_proc.returncode != 0:
            raise RuntimeError(f"Failed to install llama.cpp: {stderr.decode()}")

        # Find llama-server again
        llama_server = shutil.which("llama-server")
        if not llama_server:
            # Try common Homebrew paths
            for path in ["/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"]:
                if Path(path).exists():
                    llama_server = path
                    break

        if not llama_server:
            raise RuntimeError("llama.cpp installed but llama-server not found in PATH")

        logger.info("llama.cpp installed successfully")
        return llama_server

    async def _start_llama_cpp(
        self,
        process_id: str,
        model_id: str,
        port: int,
        **kwargs,
    ) -> NativeProcess:
        """Start llama.cpp server with Metal acceleration.

        llama.cpp provides OpenAI-compatible API via llama-server.
        Automatically installs llama.cpp and converts models if needed.
        """
        # Ensure llama.cpp is installed (auto-install if needed)
        llama_server = await self._ensure_llama_cpp_installed()

        effective_model_path = model_id

        # Check if model_id is a GGUF file path
        if not model_id.endswith(".gguf"):
            # Check for cached GGUF conversion
            cached = self._converter.get_cached_model(model_id, "gguf")
            if cached:
                logger.info(f"Using cached GGUF model: {cached}")
                effective_model_path = cached
            else:
                # Check if model has GGUF files on HuggingFace
                # For now, attempt conversion
                logger.info(f"Converting {model_id} to GGUF format...")
                try:
                    quant_type = kwargs.pop("gguf_quant", "q8_0")
                    effective_model_path = await self._converter.convert_to_gguf(
                        model_id, quant_type=quant_type
                    )
                except Exception as e:
                    logger.error(f"GGUF conversion failed: {e}")
                    raise RuntimeError(
                        f"Failed to convert model to GGUF format: {e}. "
                        "Consider using a pre-quantized GGUF model or Ollama backend."
                    )

        # Build command
        cmd = [
            llama_server,
            "--model",
            effective_model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "-ngl",
            "99",  # Offload all layers to GPU (Metal)
        ]

        # Add optional parameters
        if ctx_size := kwargs.get("ctx_size"):
            cmd.extend(["-c", str(ctx_size)])

        if n_threads := kwargs.get("n_threads"):
            cmd.extend(["-t", str(n_threads)])

        # Create log file
        log_file = self._log_dir / f"{process_id}.log"

        # Start the process with log file
        env = os.environ.copy()
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

        logger.info(f"Started llama.cpp server (PID {process.pid}) for {effective_model_path}")

        return NativeProcess(
            process_id=process_id,
            pid=process.pid,
            backend="llama_cpp",
            model_id=effective_model_path,
            port=port,
            process=process,
            log_file=log_file,
        )

    async def _ensure_vllm_metal_installed(self) -> str:
        """Ensure vLLM-Metal is installed in a virtual environment.

        Creates a virtual environment at ~/.lmstack/venvs/vllm-metal
        and installs vllm-metal if not already present.

        Returns:
            Path to the vllm command in the virtual environment
        """
        venv_dir = Path.home() / ".lmstack" / "venvs" / "vllm-metal"
        vllm_cmd = venv_dir / "bin" / "vllm"

        # Check if vllm is already installed in venv
        if vllm_cmd.exists():
            logger.info(f"vLLM-Metal already installed at {vllm_cmd}")
            return str(vllm_cmd)

        # Create virtual environment
        logger.info(f"Creating virtual environment for vLLM-Metal at {venv_dir}")
        venv_dir.parent.mkdir(parents=True, exist_ok=True)

        # Create venv
        create_venv = await asyncio.create_subprocess_exec(
            "python3",
            "-m",
            "venv",
            str(venv_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await create_venv.wait()

        if create_venv.returncode != 0:
            stderr = await create_venv.stderr.read()
            raise RuntimeError(f"Failed to create virtual environment: {stderr.decode()}")

        # Install vllm-metal
        pip_cmd = venv_dir / "bin" / "pip"
        logger.info("Installing vllm-metal (this may take a few minutes)...")

        install_proc = await asyncio.create_subprocess_exec(
            str(pip_cmd),
            "install",
            "vllm-metal",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await install_proc.communicate()

        if install_proc.returncode != 0:
            raise RuntimeError(
                f"Failed to install vllm-metal: {stderr.decode()}\n"
                "You may need to install it manually: pip install vllm-metal"
            )

        logger.info("vLLM-Metal installed successfully")
        return str(vllm_cmd)

    async def _start_vllm_metal(
        self,
        process_id: str,
        model_id: str,
        port: int,
        **kwargs,
    ) -> NativeProcess:
        """Start vLLM-Metal server for Apple Silicon.

        vLLM-Metal provides OpenAI-compatible API via `vllm serve`.
        Automatically installs vllm-metal in a virtual environment if needed.
        See: https://github.com/vllm-project/vllm-metal
        """
        # Ensure vLLM-Metal is installed (auto-install if needed)
        vllm_cmd = await self._ensure_vllm_metal_installed()

        # Build command using vllm serve
        cmd = [
            vllm_cmd,
            "serve",
            model_id,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]

        # Add optional parameters
        if gpu_memory_util := kwargs.get("gpu_memory_utilization"):
            cmd.extend(["--gpu-memory-utilization", str(gpu_memory_util)])

        if max_model_len := kwargs.get("max_model_len"):
            cmd.extend(["--max-model-len", str(max_model_len)])

        if dtype := kwargs.get("dtype"):
            cmd.extend(["--dtype", str(dtype)])

        if kwargs.get("trust_remote_code"):
            cmd.append("--trust-remote-code")

        # Create log file
        log_file = self._log_dir / f"{process_id}.log"

        # Start the process with log file
        env = os.environ.copy()
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

        logger.info(f"Started vLLM-Metal server (PID {process.pid}) for {model_id}")

        return NativeProcess(
            process_id=process_id,
            pid=process.pid,
            backend="vllm",
            model_id=model_id,
            port=port,
            process=process,
            log_file=log_file,
        )

    def get_logs(self, process_id: str, tail: int = 100) -> str:
        """Get logs from a process.

        Reads from log file for MLX, llama.cpp, and vLLM-Metal backends.
        For Ollama, returns status information about loaded models.
        """
        process = self._processes.get(process_id)
        if not process:
            return "Process not found"

        if process.backend == "ollama":
            return self._get_ollama_status(process)

        # Read from log file
        if process.log_file and process.log_file.exists():
            try:
                with open(process.log_file) as f:
                    lines = f.readlines()
                    # Return last 'tail' lines
                    if len(lines) > tail:
                        lines = lines[-tail:]
                    return "".join(lines)
            except Exception as e:
                return f"Error reading logs: {e}"

        return "No logs available"

    def _get_ollama_status(self, process: NativeProcess) -> str:
        """Get Ollama status information."""
        import httpx

        lines = []
        lines.append("=== Ollama Native Deployment ===")
        lines.append(f"Model: {process.model_id}")
        lines.append(f"Port: {process.port}")
        lines.append("")

        try:
            # Get running models
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"http://localhost:{process.port}/api/ps")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    if models:
                        lines.append("=== Loaded Models ===")
                        for m in models:
                            name = m.get("name", "unknown")
                            size = m.get("size", 0)
                            size_gb = size / (1024**3) if size else 0
                            lines.append(f"  - {name} ({size_gb:.1f} GB)")
                    else:
                        lines.append("No models currently loaded in memory")
                    lines.append("")

                # Get available models
                response = client.get(f"http://localhost:{process.port}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    if models:
                        lines.append("=== Available Models ===")
                        for m in models:
                            name = m.get("name", "unknown")
                            size = m.get("size", 0)
                            size_gb = size / (1024**3) if size else 0
                            lines.append(f"  - {name} ({size_gb:.1f} GB)")

        except Exception as e:
            lines.append(f"Error getting Ollama status: {e}")

        lines.append("")
        lines.append("Note: Detailed logs available in Console.app (macOS)")

        return "\n".join(lines)

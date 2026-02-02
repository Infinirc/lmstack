"""Native process manager for Mac workers.

Manages LLM inference processes without Docker for macOS with Apple Silicon.
Supports Ollama, MLX-LM, and llama.cpp backends.
"""

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from .converter import ModelConverter

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_PORT = 11434


@dataclass
class NativeProcess:
    """Represents a native process deployment."""

    process_id: str  # Unique identifier (deployment_id based)
    pid: int  # OS process ID
    backend: str  # ollama, mlx, llama_cpp
    model_id: str
    port: int
    process: Optional[subprocess.Popen] = None


class NativeProcessManager:
    """Manages native processes for Mac LLM deployments."""

    def __init__(self):
        self._processes: dict[str, NativeProcess] = {}
        self._ollama_process: Optional[subprocess.Popen] = None
        self._converter = ModelConverter()

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

    async def _start_mlx(
        self,
        process_id: str,
        model_id: str,
        port: int,
        **kwargs,
    ) -> NativeProcess:
        """Start MLX-LM server for Apple Silicon.

        MLX-LM provides OpenAI-compatible API via mlx_lm.server.
        Automatically converts HuggingFace models to MLX format if needed.
        """
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

        # Build command
        cmd = [
            "python3",
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

        # Start the process
        env = os.environ.copy()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
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
        )

    async def _start_llama_cpp(
        self,
        process_id: str,
        model_id: str,
        port: int,
        **kwargs,
    ) -> NativeProcess:
        """Start llama.cpp server with Metal acceleration.

        llama.cpp provides OpenAI-compatible API via llama-server.
        Automatically converts HuggingFace models to GGUF format if needed.
        """
        # Check for llama-server binary
        import shutil

        llama_server = shutil.which("llama-server")

        if not llama_server:
            raise RuntimeError(
                "llama-server not found. " "Please install llama.cpp: brew install llama.cpp"
            )

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

        # Start the process
        env = os.environ.copy()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
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
        )

    def get_logs(self, process_id: str, tail: int = 100) -> str:
        """Get logs from a process.

        For subprocess-based backends, reads from stdout pipe.
        For Ollama, returns status information about loaded models.
        """
        process = self._processes.get(process_id)
        if not process:
            return "Process not found"

        if process.backend == "ollama":
            return self._get_ollama_status(process)

        if process.process and process.process.stdout:
            try:
                # This is a simple implementation - in production you'd want
                # to capture logs to a file and read the tail
                return "Logs are available but streaming is not yet implemented"
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

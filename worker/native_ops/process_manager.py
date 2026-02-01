"""Native process manager for Mac workers.

Manages LLM inference processes without Docker for macOS with Apple Silicon.
Supports Ollama, MLX-LM, and llama.cpp backends.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


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
            raise ValueError(f"Process {process_id} already exists")

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

        ollama_port = 11434  # Ollama's default port

        # Check if Ollama service is running
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{ollama_port}/api/tags")
                if response.status_code != 200:
                    raise RuntimeError("Ollama service is not responding")
        except httpx.ConnectError:
            raise RuntimeError(
                "Ollama service is not running. " "Please start it with: ollama serve"
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
        """
        # Build command
        cmd = [
            "python3",
            "-m",
            "mlx_lm.server",
            "--model",
            model_id,
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

        logger.info(f"Started MLX-LM server (PID {process.pid}) for {model_id}")

        return NativeProcess(
            process_id=process_id,
            pid=process.pid,
            backend="mlx",
            model_id=model_id,
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
        """
        # Check for llama-server binary
        import shutil

        llama_server = shutil.which("llama-server")

        if not llama_server:
            raise RuntimeError(
                "llama-server not found. " "Please install llama.cpp: brew install llama.cpp"
            )

        # Build command
        cmd = [
            llama_server,
            "--model",
            model_id,
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

        logger.info(f"Started llama.cpp server (PID {process.pid}) for {model_id}")

        return NativeProcess(
            process_id=process_id,
            pid=process.pid,
            backend="llama_cpp",
            model_id=model_id,
            port=port,
            process=process,
        )

    def get_logs(self, process_id: str, tail: int = 100) -> str:
        """Get logs from a process.

        For subprocess-based backends, reads from stdout pipe.
        For Ollama, returns a message about checking system logs.
        """
        process = self._processes.get(process_id)
        if not process:
            return "Process not found"

        if process.backend == "ollama":
            return (
                "Ollama logs are managed by the system service.\n"
                "Check with: journalctl -u ollama (Linux) or Console.app (macOS)"
            )

        if process.process and process.process.stdout:
            try:
                # This is a simple implementation - in production you'd want
                # to capture logs to a file and read the tail
                return "Logs are available but streaming is not yet implemented"
            except Exception as e:
                return f"Error reading logs: {e}"

        return "No logs available"

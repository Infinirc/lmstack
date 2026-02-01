"""MLX-LM native backend manager for Mac.

MLX-LM is Apple's machine learning framework optimized for Apple Silicon.
It provides efficient LLM inference using the Metal GPU framework.

Usage:
    mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class MLXProcess:
    """MLX-LM server process."""

    pid: int
    model_id: str
    port: int
    process: subprocess.Popen


class MLXManager:
    """Manager for MLX-LM server processes."""

    def __init__(self):
        self._processes: dict[int, MLXProcess] = {}  # port -> process

    @staticmethod
    def is_available() -> bool:
        """Check if MLX-LM is installed."""
        try:
            result = subprocess.run(
                ["python3", "-c", "import mlx_lm; print('ok')"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except Exception:
            return False

    async def start_server(
        self,
        model_id: str,
        port: int,
        trust_remote_code: bool = False,
        quantize: Optional[str] = None,
    ) -> MLXProcess:
        """Start an MLX-LM server.

        Args:
            model_id: HuggingFace model ID or local path
                     (e.g., 'mlx-community/Llama-3.2-3B-Instruct-4bit')
            port: Port to serve on
            trust_remote_code: Trust remote code in model
            quantize: Quantization level (e.g., '4bit', '8bit')

        Returns:
            MLXProcess instance
        """
        if port in self._processes:
            raise ValueError(f"Port {port} already in use")

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

        if trust_remote_code:
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

        mlx_process = MLXProcess(
            pid=process.pid,
            model_id=model_id,
            port=port,
            process=process,
        )
        self._processes[port] = mlx_process

        logger.info(f"Started MLX-LM server (PID {process.pid}) on port {port}")

        # Wait for server to be ready
        ready = await self._wait_for_ready(port, timeout=300)
        if not ready:
            self.stop_server(port)
            raise RuntimeError(f"MLX-LM server failed to start on port {port}")

        return mlx_process

    async def _wait_for_ready(self, port: int, timeout: int = 300) -> bool:
        """Wait for MLX-LM server to be ready."""
        elapsed = 0
        check_interval = 2

        while elapsed < timeout:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"http://localhost:{port}/v1/models")
                    if response.status_code == 200:
                        logger.info(f"MLX-LM server ready on port {port}")
                        return True
            except Exception:
                pass

            await asyncio.sleep(check_interval)
            elapsed += check_interval

        return False

    def stop_server(self, port: int) -> bool:
        """Stop an MLX-LM server.

        Args:
            port: Port of the server to stop

        Returns:
            True if stopped successfully
        """
        mlx_process = self._processes.get(port)
        if not mlx_process:
            return False

        try:
            if mlx_process.process.poll() is None:
                mlx_process.process.terminate()
                try:
                    mlx_process.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    mlx_process.process.kill()

            del self._processes[port]
            logger.info(f"Stopped MLX-LM server on port {port}")
            return True

        except Exception as e:
            logger.error(f"Error stopping MLX-LM server: {e}")
            return False

    def is_running(self, port: int) -> bool:
        """Check if a server is running on a port."""
        mlx_process = self._processes.get(port)
        if not mlx_process:
            return False
        return mlx_process.process.poll() is None

    def list_servers(self) -> list[MLXProcess]:
        """List all running MLX-LM servers."""
        return [p for p in self._processes.values() if p.process.poll() is None]

    @staticmethod
    def get_openai_base_url(port: int) -> str:
        """Get OpenAI-compatible API base URL for a server."""
        return f"http://localhost:{port}/v1"

    @staticmethod
    def list_mlx_models() -> list[str]:
        """List recommended MLX models from mlx-community.

        Returns common MLX-optimized models that work well on Apple Silicon.
        """
        return [
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "mlx-community/Llama-3.1-8B-Instruct-4bit",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "mlx-community/Qwen2.5-14B-Instruct-4bit",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "mlx-community/gemma-2-9b-it-4bit",
            "mlx-community/Phi-3.5-mini-instruct-4bit",
        ]

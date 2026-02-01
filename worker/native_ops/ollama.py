"""Ollama native backend manager for Mac.

Manages Ollama service running as a native system service on macOS.
Ollama provides OpenAI-compatible API at http://localhost:11434/v1/
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_PORT = 11434


class OllamaManager:
    """Manager for native Ollama service."""

    def __init__(self, port: int = OLLAMA_DEFAULT_PORT):
        self.port = port
        self.base_url = f"http://localhost:{port}"

    async def is_running(self) -> bool:
        """Check if Ollama service is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict]:
        """List available models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])
                return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def pull_model(
        self,
        model_id: str,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """Pull a model from Ollama registry.

        Args:
            model_id: Model name (e.g., 'llama3.2', 'qwen2.5:7b')
            progress_callback: Optional callback for progress updates

        Returns:
            True if pull succeeded
        """
        try:
            async with httpx.AsyncClient(timeout=1800.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model_id, "stream": True},
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"Ollama pull failed: {response.status_code}")
                        return False

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                import json

                                data = json.loads(line)
                                status = data.get("status", "")

                                if progress_callback:
                                    progress = 0
                                    if "completed" in data and "total" in data:
                                        progress = int(data["completed"] / data["total"] * 100)
                                    progress_callback(status, progress)

                                if status == "success":
                                    return True

                            except Exception:
                                pass

                    return True

        except Exception as e:
            logger.error(f"Failed to pull Ollama model: {e}")
            return False

    async def load_model(self, model_id: str) -> bool:
        """Pre-load a model into memory.

        Ollama lazy-loads models on first request.
        This sends a dummy request to load the model.
        """
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_id,
                        "prompt": "Hello",
                        "stream": False,
                    },
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Failed to preload model: {e}")
            return False

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_id,
                        "prompt": "",
                        "keep_alive": 0,
                    },
                )
                return True
        except Exception as e:
            logger.warning(f"Failed to unload model: {e}")
            return False

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model from local storage."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.base_url}/api/delete",
                    json={"name": model_id},
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False

    def get_openai_base_url(self) -> str:
        """Get OpenAI-compatible API base URL."""
        return f"{self.base_url}/v1"

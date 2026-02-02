"""Model format converter for MLX and GGUF formats.

This module handles converting HuggingFace models to formats
compatible with MLX and llama.cpp backends.
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".lmstack" / "converted_models"


@dataclass
class ConversionTask:
    """Represents an ongoing conversion task."""

    task_id: str
    hf_model_id: str
    target_format: str  # "mlx" or "gguf"
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ModelConverter:
    """Model format converter for MLX and GGUF formats.

    Handles:
    - Converting HuggingFace models to MLX format
    - Converting HuggingFace models to GGUF format
    - Caching converted models
    - Checking if models are already in compatible formats
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, ConversionTask] = {}

    def get_mlx_cache_path(self, hf_model_id: str) -> Path:
        """Get the cache path for MLX converted model."""
        safe_name = hf_model_id.replace("/", "--")
        return self.cache_dir / "mlx" / safe_name

    def get_gguf_cache_path(self, hf_model_id: str, quant_type: str = "q8_0") -> Path:
        """Get the cache path for GGUF converted model."""
        safe_name = hf_model_id.replace("/", "--")
        return self.cache_dir / "gguf" / f"{safe_name}-{quant_type}.gguf"

    def get_cached_model(self, hf_model_id: str, format: str) -> Optional[str]:
        """Get the path to a cached converted model if it exists.

        Args:
            hf_model_id: HuggingFace model ID
            format: Target format ("mlx" or "gguf")

        Returns:
            Path to cached model if exists, None otherwise
        """
        if format == "mlx":
            cache_path = self.get_mlx_cache_path(hf_model_id)
            # MLX models are directories with config.json and model files
            if cache_path.exists() and (cache_path / "config.json").exists():
                return str(cache_path)
        elif format == "gguf":
            # Try common quantization types
            for quant in ["q8_0", "q4_k_m", "q4_0", "f16"]:
                cache_path = self.get_gguf_cache_path(hf_model_id, quant)
                if cache_path.exists():
                    return str(cache_path)
        return None

    @staticmethod
    def is_mlx_ready(model_id: str) -> bool:
        """Check if model is from mlx-community (already MLX format).

        Args:
            model_id: HuggingFace model ID

        Returns:
            True if model is from mlx-community organization
        """
        return model_id.startswith("mlx-community/")

    @staticmethod
    def is_gguf_ready(model_id: str, files: Optional[list[str]] = None) -> bool:
        """Check if model has GGUF files available.

        Args:
            model_id: HuggingFace model ID
            files: List of file names in the model repository

        Returns:
            True if model has .gguf files
        """
        if files:
            return any(f.endswith(".gguf") for f in files)
        # Common patterns for GGUF models
        return any(pattern in model_id.lower() for pattern in ["gguf", "-gguf", "_gguf"])

    async def download_gguf_model(self, hf_model_id: str) -> str:
        """Download a GGUF model from HuggingFace.

        Uses huggingface_hub to download the .gguf file(s) from a repo.

        Args:
            hf_model_id: HuggingFace model ID (e.g., "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF")

        Returns:
            Path to downloaded GGUF file
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

        # Create cache directory for downloaded models
        cache_dir = self.cache_dir / "gguf"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # List files in the repo to find .gguf files
        try:
            files = list_repo_files(hf_model_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]

            if not gguf_files:
                raise RuntimeError(f"No .gguf files found in {hf_model_id}")

            # Pick the best file (prefer Q8_0 or largest quantization)
            gguf_file = gguf_files[0]
            for f in gguf_files:
                # Prefer Q8_0 quantization
                if "q8_0" in f.lower() or "Q8_0" in f:
                    gguf_file = f
                    break

            logger.info(f"Downloading {gguf_file} from {hf_model_id}...")

            # Download the file
            local_path = hf_hub_download(
                repo_id=hf_model_id,
                filename=gguf_file,
                cache_dir=str(cache_dir),
                local_dir=str(cache_dir / hf_model_id.replace("/", "--")),
                local_dir_use_symlinks=False,
            )

            logger.info(f"Downloaded GGUF model to {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download GGUF model: {e}")
            raise RuntimeError(f"Failed to download GGUF model from {hf_model_id}: {e}")

    @staticmethod
    def find_mlx_variant(hf_model_id: str) -> Optional[str]:
        """Find MLX variant of a HuggingFace model.

        Searches mlx-community for a converted version of the model.

        Args:
            hf_model_id: Original HuggingFace model ID

        Returns:
            MLX model ID if found, None otherwise
        """
        # Try common naming patterns
        model_name = hf_model_id.split("/")[-1]
        patterns = [
            f"mlx-community/{model_name}",
            f"mlx-community/{model_name}-mlx",
            f"mlx-community/{model_name}-4bit",
            f"mlx-community/{model_name}-8bit",
        ]
        return patterns[0] if patterns else None

    def get_task(self, task_id: str) -> Optional[ConversionTask]:
        """Get conversion task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[ConversionTask]:
        """List all conversion tasks."""
        return list(self._tasks.values())

    async def convert_to_mlx(
        self,
        hf_model_id: str,
        quantize: bool = True,
        bits: int = 4,
    ) -> str:
        """Convert a HuggingFace model to MLX format.

        Uses mlx_lm.convert to convert the model.

        Args:
            hf_model_id: HuggingFace model ID
            quantize: Whether to quantize the model
            bits: Quantization bits (4 or 8)

        Returns:
            Path to converted model

        Raises:
            RuntimeError: If conversion fails
        """
        # Check if already cached
        cached = self.get_cached_model(hf_model_id, "mlx")
        if cached:
            logger.info(f"Using cached MLX model: {cached}")
            return cached

        # Check if already MLX format
        if self.is_mlx_ready(hf_model_id):
            logger.info(f"Model {hf_model_id} is already MLX format")
            return hf_model_id

        output_path = self.get_mlx_cache_path(hf_model_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create task
        task_id = f"mlx-{hf_model_id.replace('/', '--')}"
        task = ConversionTask(
            task_id=task_id,
            hf_model_id=hf_model_id,
            target_format="mlx",
            status="running",
            progress=0.0,
            message="Starting MLX conversion...",
            started_at=datetime.now(),
        )
        self._tasks[task_id] = task

        try:
            # Check if mlx_lm is available
            mlx_convert = shutil.which("mlx_lm.convert")
            if not mlx_convert:
                # Try using python module
                cmd = [
                    "python3",
                    "-m",
                    "mlx_lm.convert",
                    "--hf-path",
                    hf_model_id,
                    "--mlx-path",
                    str(output_path),
                ]
            else:
                cmd = [
                    mlx_convert,
                    "--hf-path",
                    hf_model_id,
                    "--mlx-path",
                    str(output_path),
                ]

            if quantize:
                cmd.extend(["-q", "--q-bits", str(bits)])

            task.progress = 0.1
            task.message = f"Converting {hf_model_id} to MLX format..."
            logger.info(f"Running: {' '.join(cmd)}")

            # Run conversion
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            task.progress = 0.5
            stdout, _ = await process.communicate()

            if process.returncode != 0:
                error_msg = stdout.decode() if stdout else "Unknown error"
                raise RuntimeError(f"MLX conversion failed: {error_msg}")

            task.progress = 1.0
            task.status = "completed"
            task.message = "Conversion completed"
            task.output_path = str(output_path)
            task.completed_at = datetime.now()

            logger.info(f"MLX conversion completed: {output_path}")
            return str(output_path)

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.message = f"Conversion failed: {e}"
            logger.error(f"MLX conversion failed for {hf_model_id}: {e}")
            raise

    async def convert_to_gguf(
        self,
        hf_model_id: str,
        quant_type: str = "q8_0",
    ) -> str:
        """Convert a HuggingFace model to GGUF format.

        Uses llama.cpp's convert scripts to create GGUF.

        Args:
            hf_model_id: HuggingFace model ID
            quant_type: Quantization type (q4_0, q4_k_m, q8_0, f16, etc.)

        Returns:
            Path to converted model

        Raises:
            RuntimeError: If conversion fails
        """
        # Check if already cached
        cached = self.get_cached_model(hf_model_id, "gguf")
        if cached:
            logger.info(f"Using cached GGUF model: {cached}")
            return cached

        output_path = self.get_gguf_cache_path(hf_model_id, quant_type)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create task
        task_id = f"gguf-{hf_model_id.replace('/', '--')}"
        task = ConversionTask(
            task_id=task_id,
            hf_model_id=hf_model_id,
            target_format="gguf",
            status="running",
            progress=0.0,
            message="Starting GGUF conversion...",
            started_at=datetime.now(),
        )
        self._tasks[task_id] = task

        try:
            # First, download the model using huggingface-cli
            task.progress = 0.1
            task.message = f"Downloading {hf_model_id}..."

            hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir = hf_cache_dir / f"models--{hf_model_id.replace('/', '--')}"

            if not model_dir.exists():
                download_cmd = [
                    "huggingface-cli",
                    "download",
                    hf_model_id,
                    "--local-dir",
                    str(self.cache_dir / "downloads" / hf_model_id.replace("/", "--")),
                ]
                process = await asyncio.create_subprocess_exec(
                    *download_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                stdout, _ = await process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(
                        f"Download failed: {stdout.decode() if stdout else 'Unknown error'}"
                    )
                model_dir = self.cache_dir / "downloads" / hf_model_id.replace("/", "--")

            task.progress = 0.4
            task.message = "Converting to GGUF..."

            # Find llama.cpp convert script
            # Common locations
            convert_script = None
            for path in [
                shutil.which("convert_hf_to_gguf.py"),
                Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
                Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
            ]:
                if path and Path(path).exists():
                    convert_script = str(path)
                    break

            if not convert_script:
                # Try using llama-quantize directly if model is already GGUF
                raise RuntimeError(
                    "llama.cpp convert script not found. "
                    "Please install llama.cpp: brew install llama.cpp"
                )

            # Convert to GGUF
            temp_gguf = output_path.parent / f"{output_path.stem}_temp.gguf"
            convert_cmd = [
                "python3",
                convert_script,
                str(model_dir),
                "--outfile",
                str(temp_gguf),
                "--outtype",
                "f16",
            ]

            process = await asyncio.create_subprocess_exec(
                *convert_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(
                    f"GGUF conversion failed: {stdout.decode() if stdout else 'Unknown error'}"
                )

            task.progress = 0.7
            task.message = f"Quantizing to {quant_type}..."

            # Quantize if needed
            if quant_type != "f16":
                llama_quantize = shutil.which("llama-quantize")
                if not llama_quantize:
                    raise RuntimeError("llama-quantize not found. Please install llama.cpp")

                quant_cmd = [
                    llama_quantize,
                    str(temp_gguf),
                    str(output_path),
                    quant_type.upper(),
                ]
                process = await asyncio.create_subprocess_exec(
                    *quant_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await process.communicate()

                # Remove temp file
                temp_gguf.unlink(missing_ok=True)
            else:
                # Just rename
                temp_gguf.rename(output_path)

            task.progress = 1.0
            task.status = "completed"
            task.message = "Conversion completed"
            task.output_path = str(output_path)
            task.completed_at = datetime.now()

            logger.info(f"GGUF conversion completed: {output_path}")
            return str(output_path)

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.message = f"Conversion failed: {e}"
            logger.error(f"GGUF conversion failed for {hf_model_id}: {e}")
            raise

    def clear_cache(self, hf_model_id: Optional[str] = None, format: Optional[str] = None):
        """Clear converted model cache.

        Args:
            hf_model_id: Clear cache for specific model (None = all)
            format: Clear cache for specific format (None = all)
        """
        if hf_model_id:
            if format in (None, "mlx"):
                cache_path = self.get_mlx_cache_path(hf_model_id)
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                    logger.info(f"Cleared MLX cache: {cache_path}")

            if format in (None, "gguf"):
                for quant in ["q8_0", "q4_k_m", "q4_0", "f16"]:
                    cache_path = self.get_gguf_cache_path(hf_model_id, quant)
                    if cache_path.exists():
                        cache_path.unlink()
                        logger.info(f"Cleared GGUF cache: {cache_path}")
        else:
            # Clear all
            if format in (None, "mlx"):
                mlx_dir = self.cache_dir / "mlx"
                if mlx_dir.exists():
                    shutil.rmtree(mlx_dir)
                    logger.info("Cleared all MLX cache")

            if format in (None, "gguf"):
                gguf_dir = self.cache_dir / "gguf"
                if gguf_dir.exists():
                    shutil.rmtree(gguf_dir)
                    logger.info("Cleared all GGUF cache")

    def get_cache_info(self) -> dict:
        """Get information about cached models.

        Returns:
            Dictionary with cache statistics
        """
        info = {
            "cache_dir": str(self.cache_dir),
            "mlx_models": [],
            "gguf_models": [],
            "total_size_bytes": 0,
        }

        mlx_dir = self.cache_dir / "mlx"
        if mlx_dir.exists():
            for model_dir in mlx_dir.iterdir():
                if model_dir.is_dir():
                    size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                    info["mlx_models"].append(
                        {
                            "model_id": model_dir.name.replace("--", "/"),
                            "path": str(model_dir),
                            "size_bytes": size,
                        }
                    )
                    info["total_size_bytes"] += size

        gguf_dir = self.cache_dir / "gguf"
        if gguf_dir.exists():
            for gguf_file in gguf_dir.glob("*.gguf"):
                size = gguf_file.stat().st_size
                # Parse model name from filename (name-quant.gguf)
                parts = gguf_file.stem.rsplit("-", 1)
                model_id = parts[0].replace("--", "/") if parts else gguf_file.stem
                quant = parts[1] if len(parts) > 1 else "unknown"
                info["gguf_models"].append(
                    {
                        "model_id": model_id,
                        "quant_type": quant,
                        "path": str(gguf_file),
                        "size_bytes": size,
                    }
                )
                info["total_size_bytes"] += size

        return info

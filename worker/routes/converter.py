"""Model conversion routes for the worker agent.

Provides API endpoints for converting HuggingFace models to MLX/GGUF formats.
"""

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from worker.agent import WorkerAgent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["converter"])

# Global agent reference (set by agent.py)
_agent: "WorkerAgent | None" = None


def set_agent(agent: "WorkerAgent"):
    """Set the global agent reference."""
    global _agent
    _agent = agent


def _get_converter():
    """Get the converter from native manager."""
    if not _agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    if not hasattr(_agent, "native_manager") or not _agent.native_manager:
        raise HTTPException(
            status_code=400,
            detail="Model conversion only available on Mac workers with native support",
        )
    return _agent.native_manager._converter


class MLXConvertRequest(BaseModel):
    """Request to convert a model to MLX format."""

    hf_model_id: str
    quantize: bool = True
    bits: int = 4  # 4 or 8


class GGUFConvertRequest(BaseModel):
    """Request to convert a model to GGUF format."""

    hf_model_id: str
    quant_type: str = "q8_0"  # q4_0, q4_k_m, q8_0, f16


class ConvertResponse(BaseModel):
    """Response from conversion request."""

    task_id: str
    status: str
    message: str
    output_path: Optional[str] = None


class ConversionProgress(BaseModel):
    """Conversion task progress."""

    task_id: str
    status: str
    progress: float
    message: str
    output_path: Optional[str] = None
    error: Optional[str] = None


class FormatCheckRequest(BaseModel):
    """Request to check model format compatibility."""

    model_id: str
    files: Optional[list[str]] = None


class FormatCheckResponse(BaseModel):
    """Response with model format compatibility info."""

    model_id: str
    is_mlx_ready: bool
    is_gguf_ready: bool
    cached_mlx: Optional[str] = None
    cached_gguf: Optional[str] = None


@router.post("/convert/mlx", response_model=ConvertResponse)
async def convert_to_mlx(request: MLXConvertRequest):
    """Convert a HuggingFace model to MLX format.

    This endpoint starts the conversion process and returns immediately.
    Use GET /convert/progress/{task_id} to check progress.
    """
    converter = _get_converter()

    try:
        # Start conversion
        output_path = await converter.convert_to_mlx(
            hf_model_id=request.hf_model_id,
            quantize=request.quantize,
            bits=request.bits,
        )

        task_id = f"mlx-{request.hf_model_id.replace('/', '--')}"

        return ConvertResponse(
            task_id=task_id,
            status="completed",
            message="Conversion completed successfully",
            output_path=output_path,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"MLX conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/convert/gguf", response_model=ConvertResponse)
async def convert_to_gguf(request: GGUFConvertRequest):
    """Convert a HuggingFace model to GGUF format.

    This endpoint starts the conversion process and returns immediately.
    Use GET /convert/progress/{task_id} to check progress.
    """
    converter = _get_converter()

    try:
        output_path = await converter.convert_to_gguf(
            hf_model_id=request.hf_model_id,
            quant_type=request.quant_type,
        )

        task_id = f"gguf-{request.hf_model_id.replace('/', '--')}"

        return ConvertResponse(
            task_id=task_id,
            status="completed",
            message="Conversion completed successfully",
            output_path=output_path,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"GGUF conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/convert/progress/{task_id}", response_model=ConversionProgress)
async def get_conversion_progress(task_id: str):
    """Get the progress of a conversion task."""
    converter = _get_converter()

    task = converter.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return ConversionProgress(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        message=task.message,
        output_path=task.output_path,
        error=task.error,
    )


@router.get("/convert/tasks")
async def list_conversion_tasks():
    """List all conversion tasks."""
    converter = _get_converter()

    tasks = converter.list_tasks()
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "hf_model_id": t.hf_model_id,
                "target_format": t.target_format,
                "status": t.status,
                "progress": t.progress,
                "message": t.message,
            }
            for t in tasks
        ]
    }


@router.post("/convert/check-format", response_model=FormatCheckResponse)
async def check_model_format(request: FormatCheckRequest):
    """Check if a model is already in MLX or GGUF format."""
    converter = _get_converter()

    from worker.native_ops.converter import ModelConverter

    is_mlx = ModelConverter.is_mlx_ready(request.model_id)
    is_gguf = ModelConverter.is_gguf_ready(request.model_id, request.files)

    return FormatCheckResponse(
        model_id=request.model_id,
        is_mlx_ready=is_mlx,
        is_gguf_ready=is_gguf,
        cached_mlx=converter.get_cached_model(request.model_id, "mlx") if not is_mlx else None,
        cached_gguf=converter.get_cached_model(request.model_id, "gguf") if not is_gguf else None,
    )


@router.get("/convert/cache")
async def get_cache_info():
    """Get information about cached converted models."""
    converter = _get_converter()
    return converter.get_cache_info()


@router.delete("/convert/cache")
async def clear_cache(model_id: Optional[str] = None, format: Optional[str] = None):
    """Clear the model conversion cache.

    Args:
        model_id: Clear cache for specific model (None = all)
        format: Clear cache for specific format: "mlx" or "gguf" (None = all)
    """
    converter = _get_converter()
    converter.clear_cache(model_id, format)
    return {"status": "ok", "message": "Cache cleared"}

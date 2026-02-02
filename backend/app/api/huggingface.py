"""HuggingFace API integration for model info and VRAM estimation"""

import re

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()

# HuggingFace API base URL
HF_API_URL = "https://huggingface.co/api"

# In-memory cache for model info (simple TTL cache).
# NOTE: This is not suitable for multi-process deployments. For production
# with multiple workers, use Redis or a distributed cache.
_model_cache: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 300  # 5 minutes
_MAX_CACHE_ENTRIES = 500  # Limit to prevent memory leaks


class ModelInfo(BaseModel):
    """HuggingFace model information"""

    id: str
    model_id: str
    author: str | None = None
    sha: str | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: list[str] = []
    downloads: int = 0
    likes: int = 0
    private: bool = False
    gated: str | None = None
    created_at: str | None = None
    last_modified: str | None = None
    # Computed fields
    size_bytes: int | None = None
    parameter_count: str | None = None
    description: str | None = None


class VRAMEstimate(BaseModel):
    """VRAM estimation result"""

    model_id: str
    parameter_count: float | None = None  # in billions
    estimated_vram_gb: float
    precision: str  # fp32, fp16, bf16, int8, int4
    breakdown: dict[str, float]  # Component breakdown
    compatible: bool = True
    messages: list[str] = []


class ModelFile(BaseModel):
    """Model file information"""

    filename: str
    size: int
    type: str  # "model", "config", "tokenizer", etc.


def _get_cached(key: str) -> dict | None:
    """Get cached value if not expired"""
    import time

    if key in _model_cache:
        timestamp, data = _model_cache[key]
        if time.time() - timestamp < CACHE_TTL:
            return data
        del _model_cache[key]
    return None


def _set_cache(key: str, data: dict):
    """Set cache value with automatic cleanup of old entries."""
    import time

    current_time = time.time()

    # Cleanup expired entries if cache is getting large
    if len(_model_cache) >= _MAX_CACHE_ENTRIES:
        expired_keys = [k for k, (ts, _) in _model_cache.items() if current_time - ts >= CACHE_TTL]
        for k in expired_keys:
            _model_cache.pop(k, None)

        # If still too large, remove oldest entries
        if len(_model_cache) >= _MAX_CACHE_ENTRIES:
            sorted_keys = sorted(_model_cache.keys(), key=lambda k: _model_cache[k][0])
            for k in sorted_keys[: _MAX_CACHE_ENTRIES // 4]:
                _model_cache.pop(k, None)

    _model_cache[key] = (current_time, data)


def _parse_parameter_count(model_id: str, config: dict, tags: list[str]) -> float | None:
    """
    Parse parameter count from model config or ID.
    Returns count in billions.

    Priority:
    1. Explicit num_parameters in config (most reliable)
    2. Parse from model name (e.g., "120B", "8B")
    3. Parse from tags
    4. Estimate from architecture config (least reliable, deprecated for modern models)
    """
    # 1. Check if config has explicit num_parameters (most reliable)
    if config and "num_parameters" in config:
        return config["num_parameters"] / 1e9

    # 2. Try to parse from model ID (e.g., "Llama-3.1-8B" -> 8, "Qwen2.5-72B-Instruct" -> 72)
    # This is usually reliable for well-known models
    patterns = [
        r"[-_](\d+(?:\.\d+)?)[Bb](?:[-_]|$)",  # "-8B-", "_70B_", "-1.5B"
        r"[-_](\d+(?:\.\d+)?)[Bb]$",  # ends with "-8B"
        r"(\d+(?:\.\d+)?)[Bb][-_]",  # "8B-", "70B_"
        r"[^0-9](\d+(?:\.\d+)?)[Bb][^a-zA-Z0-9]",  # "8B" surrounded by non-alphanumeric
        r"[^0-9](\d+(?:\.\d+)?)[Bb]$",  # "8B" at end after non-digit
    ]

    model_name = model_id.split("/")[-1]
    for pattern in patterns:
        match = re.search(pattern, model_name)
        if match:
            size = float(match.group(1))
            # Sanity check: model size should be reasonable (0.1B to 2000B)
            if 0.1 <= size <= 2000:
                return size

    # 3. Check tags for size hints (e.g., "8B", "70b")
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower.endswith("b"):
            try:
                size = float(tag_lower[:-1])
                if 0.1 <= size <= 2000:  # Reasonable range
                    return size
            except ValueError:
                pass

    # 4. Estimate from architecture config (fallback, may be inaccurate for modern architectures)
    # Note: This estimation doesn't work well for MoE, GQA, or other modern architectures
    if config:
        hidden_size = config.get("hidden_size", config.get("d_model", 0))
        num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
        vocab_size = config.get("vocab_size", 0)
        intermediate_size = config.get("intermediate_size", hidden_size * 4 if hidden_size else 0)

        # Check for MoE (Mixture of Experts) models
        num_experts = config.get("num_local_experts", config.get("num_experts", 1))

        if hidden_size and num_layers and vocab_size:
            # Embedding: vocab_size * hidden_size * 2 (input + output)
            embedding = vocab_size * hidden_size * 2

            # Attention: Q, K, V, O projections
            # For GQA, K and V may be smaller, but we estimate conservatively
            num_kv_heads = config.get("num_key_value_heads", config.get("num_attention_heads", 0))
            num_attention_heads = config.get("num_attention_heads", num_kv_heads)
            head_dim = hidden_size // num_attention_heads if num_attention_heads else 0

            if head_dim and num_attention_heads and num_kv_heads:
                q_proj = hidden_size * num_attention_heads * head_dim
                k_proj = hidden_size * num_kv_heads * head_dim
                v_proj = hidden_size * num_kv_heads * head_dim
                o_proj = num_attention_heads * head_dim * hidden_size
                attention = (q_proj + k_proj + v_proj + o_proj) * num_layers
            else:
                attention = 4 * hidden_size * hidden_size * num_layers

            # FFN: gate, up, down projections (for SwiGLU style)
            # For MoE, multiply by num_experts
            ffn = 3 * hidden_size * intermediate_size * num_layers * num_experts

            # LayerNorm/RMSNorm
            layernorm = 4 * hidden_size * num_layers

            total = embedding + attention + ffn + layernorm
            return total / 1e9

    return None


def estimate_vram(
    parameter_count: float,
    precision: str = "fp16",
    context_length: int = 4096,
    batch_size: int = 1,
) -> tuple[float, dict[str, float]]:
    """
    Estimate VRAM usage for a model.

    Args:
        parameter_count: Number of parameters in billions
        precision: Model precision (fp32, fp16, bf16, int8, int4)
        context_length: Maximum context length
        batch_size: Batch size for inference

    Returns:
        Tuple of (total_vram_gb, breakdown_dict)
    """
    # Bytes per parameter based on precision
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
        "auto": 2,  # Assume fp16
    }.get(precision.lower(), 2)

    params_billions = parameter_count

    # 1. Model weights
    model_weights_gb = params_billions * bytes_per_param

    # 2. KV Cache estimation (rough)
    # Assuming typical transformer architecture
    # KV cache ≈ 2 * num_layers * hidden_size * context_length * batch_size * bytes_per_param
    # Simplified: ~0.5-2GB per 1B params for 4K context
    kv_cache_factor = context_length / 4096
    kv_cache_gb = params_billions * 0.5 * kv_cache_factor * bytes_per_param / 2

    # 3. Activation memory (during inference)
    # Usually 10-20% of model weights for inference
    activation_gb = model_weights_gb * 0.15

    # 4. CUDA/Framework overhead
    # Usually 0.5-1GB fixed overhead
    overhead_gb = 0.8

    # Total
    total_gb = model_weights_gb + kv_cache_gb + activation_gb + overhead_gb

    breakdown = {
        "model_weights": round(model_weights_gb, 2),
        "kv_cache": round(kv_cache_gb, 2),
        "activations": round(activation_gb, 2),
        "overhead": round(overhead_gb, 2),
    }

    return round(total_gb, 2), breakdown


@router.get("/model/{model_id:path}", response_model=ModelInfo)
async def get_model_info(
    model_id: str,
    token: str | None = Query(None, description="HuggingFace API token"),
):
    """
    Get model information from HuggingFace.

    The model_id should be in format "owner/model-name".
    """
    # Check cache
    cached = _get_cached(f"model:{model_id}")
    if cached:
        return ModelInfo(**cached)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get model info
            response = await client.get(
                f"{HF_API_URL}/models/{model_id}",
                headers=headers,
            )

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Model not found")
            elif response.status_code == 401:
                raise HTTPException(
                    status_code=401, detail="Invalid or missing token for gated model"
                )
            elif response.status_code == 403:
                raise HTTPException(
                    status_code=403,
                    detail="Access denied. Model may require accepting terms.",
                )

            response.raise_for_status()
            data = response.json()

            # Try to get config for parameter estimation
            config = {}
            try:
                config_response = await client.get(
                    f"https://huggingface.co/{model_id}/raw/main/config.json",
                    headers=headers,
                    timeout=10.0,
                )
                if config_response.status_code == 200:
                    config = config_response.json()
            except (httpx.RequestError, json.JSONDecodeError) as e:
                logger.debug(f"Could not fetch config for {model_id}: {e}")

            # Parse parameter count
            param_count = _parse_parameter_count(model_id, config, data.get("tags", []))

            # Calculate total size from siblings (files)
            total_size = 0
            siblings = data.get("siblings", [])
            for file in siblings:
                filename = file.get("rfilename", "")
                if any(filename.endswith(ext) for ext in [".safetensors", ".bin", ".pt", ".pth"]):
                    total_size += file.get("size", 0)

            # Handle gated field - can be False (bool), None, or string like "manual"/"auto"
            gated_value = data.get("gated")
            if isinstance(gated_value, bool):
                gated_value = None if not gated_value else "true"

            result = {
                "id": data.get("_id", model_id),
                "model_id": data.get("modelId", model_id),
                "author": data.get("author"),
                "sha": data.get("sha"),
                "pipeline_tag": data.get("pipeline_tag"),
                "library_name": data.get("library_name"),
                "tags": data.get("tags", []),
                "downloads": data.get("downloads", 0),
                "likes": data.get("likes", 0),
                "private": data.get("private", False),
                "gated": gated_value,
                "created_at": data.get("createdAt"),
                "last_modified": data.get("lastModified"),
                "size_bytes": total_size if total_size > 0 else None,
                "parameter_count": f"{param_count:.1f}B" if param_count else None,
                "description": (
                    data.get("cardData", {}).get("description")
                    if isinstance(data.get("cardData"), dict)
                    else None
                ),
            }

            _set_cache(f"model:{model_id}", result)
            return ModelInfo(**result)

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to HuggingFace: {str(e)}")


@router.get("/estimate-vram/{model_id:path}", response_model=VRAMEstimate)
async def estimate_model_vram(
    model_id: str,
    precision: str = Query("fp16", description="Model precision: fp32, fp16, bf16, int8, int4"),
    context_length: int = Query(4096, description="Context length"),
    gpu_memory_gb: float | None = Query(
        None, description="Available GPU memory in GB for compatibility check"
    ),
    token: str | None = Query(None, description="HuggingFace API token"),
):
    """
    Estimate VRAM usage for a model.

    Returns estimated VRAM consumption and compatibility status.
    """
    # First get model info
    try:
        model_info = await get_model_info(model_id, token)
    except HTTPException:
        # If we can't get model info, try to parse from ID
        param_count = _parse_parameter_count(model_id, {}, [])
        if not param_count:
            raise HTTPException(
                status_code=400,
                detail="Could not determine model size. Please ensure the model ID is correct.",
            )
        model_info = ModelInfo(
            id=model_id, model_id=model_id, parameter_count=f"{param_count:.1f}B"
        )

    # Parse parameter count
    param_count = None
    if model_info.parameter_count:
        try:
            param_count = float(model_info.parameter_count.replace("B", ""))
        except ValueError:
            pass

    if not param_count:
        # Try to estimate from file size
        if model_info.size_bytes:
            # Rough estimate: file size / 2 (fp16) = params
            param_count = model_info.size_bytes / (2 * 1e9)
        else:
            raise HTTPException(status_code=400, detail="Could not determine model parameter count")

    # Estimate VRAM
    vram_gb, breakdown = estimate_vram(
        param_count,
        precision=precision,
        context_length=context_length,
    )

    # Check compatibility
    messages = []
    compatible = True

    if gpu_memory_gb:
        if vram_gb > gpu_memory_gb:
            compatible = False
            messages.append(
                f"Model requires approximately {vram_gb:.2f} GiB VRAM, "
                f"but only {gpu_memory_gb:.2f} GiB available"
            )
        elif vram_gb > gpu_memory_gb * 0.9:
            messages.append(
                f"Model will use {(vram_gb / gpu_memory_gb * 100):.0f}% of available VRAM. "
                "Performance may be affected."
            )

    if compatible and not messages:
        messages.append(
            f"Compatibility check passed. "
            f"The model will consume approximately {vram_gb:.2f} GiB VRAM."
        )

    return VRAMEstimate(
        model_id=model_id,
        parameter_count=param_count,
        estimated_vram_gb=vram_gb,
        precision=precision,
        breakdown=breakdown,
        compatible=compatible,
        messages=messages,
    )


@router.get("/files/{model_id:path}", response_model=list[ModelFile])
async def list_model_files(
    model_id: str,
    token: str | None = Query(None, description="HuggingFace API token"),
):
    """
    List files in a HuggingFace model repository.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{HF_API_URL}/models/{model_id}",
                headers=headers,
                params={"blobs": True},
            )

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Model not found")

            response.raise_for_status()
            data = response.json()

            files = []
            for sibling in data.get("siblings", []):
                filename = sibling.get("rfilename", "")
                size = sibling.get("size", 0)

                # Determine file type
                if any(filename.endswith(ext) for ext in [".safetensors", ".bin", ".pt", ".pth"]):
                    file_type = "model"
                elif filename.endswith("config.json"):
                    file_type = "config"
                elif "tokenizer" in filename.lower():
                    file_type = "tokenizer"
                elif filename.endswith(".md"):
                    file_type = "readme"
                else:
                    file_type = "other"

                files.append(
                    ModelFile(
                        filename=filename,
                        size=size,
                        type=file_type,
                    )
                )

            return files

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to HuggingFace: {str(e)}")


@router.get("/search")
async def search_models(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Number of results"),
    filter_task: str | None = Query(None, description="Filter by task (e.g., text-generation)"),
):
    """
    Search for models on HuggingFace.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "search": query,
                "limit": limit,
                "sort": "downloads",
                "direction": -1,
            }
            if filter_task:
                params["filter"] = filter_task

            response = await client.get(
                f"{HF_API_URL}/models",
                params=params,
            )
            response.raise_for_status()

            models = response.json()
            return [
                {
                    "id": m.get("modelId", m.get("id")),
                    "author": m.get("author"),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "pipeline_tag": m.get("pipeline_tag"),
                    "tags": m.get("tags", [])[:5],  # Limit tags
                }
                for m in models
            ]

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Failed to search HuggingFace: {str(e)}")


@router.get("/popular")
async def get_popular_models(
    limit: int = Query(20, ge=1, le=50, description="Number of results"),
):
    """
    Get popular text-generation models from HuggingFace.
    Returns trending models sorted by downloads.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "filter": "text-generation",
                "sort": "downloads",
                "direction": -1,
                "limit": limit,
            }

            response = await client.get(
                f"{HF_API_URL}/models",
                params=params,
            )
            response.raise_for_status()

            models = response.json()
            return [
                {
                    "id": m.get("modelId", m.get("id")),
                    "author": m.get("author"),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "pipeline_tag": m.get("pipeline_tag"),
                    "tags": m.get("tags", [])[:5],
                }
                for m in models
            ]

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch popular models: {str(e)}")


@router.get("/readme/{model_id:path}")
async def get_model_readme(
    model_id: str,
    token: str | None = Query(None, description="HuggingFace API token"),
):
    """
    Get the README.md content for a HuggingFace model.
    Returns the raw markdown content of the model card.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch README.md from the model repo
            response = await client.get(
                f"https://huggingface.co/{model_id}/raw/main/README.md",
                headers=headers,
            )

            if response.status_code == 404:
                return {"content": None, "message": "README.md not found"}
            elif response.status_code == 401:
                # Gated model - README requires authentication
                return {
                    "content": None,
                    "message": "This is a gated model. README requires HuggingFace authentication.",
                }
            elif response.status_code == 403:
                # Access denied - need to accept terms
                return {
                    "content": None,
                    "message": "Access denied. You may need to accept the model's terms on HuggingFace.",
                }

            response.raise_for_status()
            return {"content": response.text}

    except httpx.RequestError as e:
        return {"content": None, "message": f"Failed to fetch README: {str(e)}"}


class ModelFormatInfo(BaseModel):
    """Model format compatibility information"""

    model_id: str
    is_mlx_ready: bool = False  # True if from mlx-community
    is_gguf_ready: bool = False  # True if has .gguf files
    mlx_variants: list[str] = []  # Available MLX variants
    gguf_files: list[str] = []  # Available GGUF files


def _is_mlx_ready(model_id: str) -> bool:
    """Check if model is from mlx-community."""
    return model_id.startswith("mlx-community/")


def _is_gguf_ready(files: list[str]) -> bool:
    """Check if model has GGUF files."""
    return any(f.endswith(".gguf") for f in files)


@router.get("/format-info/{model_id:path}", response_model=ModelFormatInfo)
async def get_model_format_info(
    model_id: str,
    token: str | None = Query(None, description="HuggingFace API token"),
):
    """
    Get model format compatibility information.

    Returns whether the model is MLX-ready, GGUF-ready, and lists available variants.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    result = ModelFormatInfo(
        model_id=model_id,
        is_mlx_ready=_is_mlx_ready(model_id),
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get model files to check for GGUF
            response = await client.get(
                f"{HF_API_URL}/models/{model_id}",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                siblings = data.get("siblings", [])
                files = [s.get("rfilename", "") for s in siblings]

                # Check for GGUF files
                gguf_files = [f for f in files if f.endswith(".gguf")]
                result.gguf_files = gguf_files
                result.is_gguf_ready = len(gguf_files) > 0

            # Search for MLX variants if not already MLX
            if not result.is_mlx_ready:
                model_name = model_id.split("/")[-1]
                # Search mlx-community for this model
                search_response = await client.get(
                    f"{HF_API_URL}/models",
                    params={
                        "search": model_name,
                        "author": "mlx-community",
                        "limit": 5,
                    },
                )
                if search_response.status_code == 200:
                    mlx_models = search_response.json()
                    result.mlx_variants = [m.get("modelId", m.get("id", "")) for m in mlx_models]

    except httpx.RequestError as e:
        # Log error but don't fail - return partial info
        import logging

        logging.getLogger(__name__).warning(f"Failed to fetch format info: {e}")

    return result


@router.get("/search-mlx")
async def search_mlx_models(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=50, description="Number of results"),
):
    """
    Search for MLX-ready models from mlx-community.

    Returns models that are already converted to MLX format.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "search": query,
                "author": "mlx-community",
                "limit": limit,
                "sort": "downloads",
                "direction": -1,
            }

            response = await client.get(
                f"{HF_API_URL}/models",
                params=params,
            )
            response.raise_for_status()

            models = response.json()
            return [
                {
                    "id": m.get("modelId", m.get("id")),
                    "author": m.get("author"),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "pipeline_tag": m.get("pipeline_tag"),
                    "tags": m.get("tags", [])[:5],
                    "is_mlx_ready": True,
                }
                for m in models
            ]

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Failed to search HuggingFace: {str(e)}")


@router.get("/search-gguf")
async def search_gguf_models(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=50, description="Number of results"),
):
    """
    Search for models with GGUF files available.

    Returns models that have pre-converted GGUF files.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Search with GGUF tag
            params = {
                "search": query,
                "limit": limit * 2,  # Get more to filter
                "sort": "downloads",
                "direction": -1,
                "filter": "gguf",
            }

            response = await client.get(
                f"{HF_API_URL}/models",
                params=params,
            )
            response.raise_for_status()

            models = response.json()

            # Filter to only include models with GGUF in name or tags
            gguf_models = []
            for m in models:
                model_id = m.get("modelId", m.get("id", ""))
                tags = m.get("tags", [])

                # Check if model has GGUF indicator
                is_gguf = "gguf" in model_id.lower() or any("gguf" in t.lower() for t in tags)

                if is_gguf:
                    gguf_models.append(
                        {
                            "id": model_id,
                            "author": m.get("author"),
                            "downloads": m.get("downloads", 0),
                            "likes": m.get("likes", 0),
                            "pipeline_tag": m.get("pipeline_tag"),
                            "tags": tags[:5],
                            "is_gguf_ready": True,
                        }
                    )

                if len(gguf_models) >= limit:
                    break

            return gguf_models

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Failed to search HuggingFace: {str(e)}")

"""Ollama library API - browse available models from ollama.com"""

import re
import time

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()

# Ollama library URL
OLLAMA_LIBRARY_URL = "https://ollama.com/library"
OLLAMA_SEARCH_URL = "https://ollama.com/search"

# In-memory cache for scraped models.
# NOTE: This is not suitable for multi-process deployments. For production
# with multiple workers, use Redis or a distributed cache.
_models_cache: dict[str, tuple[float, list]] = {}
CACHE_TTL = 600  # 10 minutes
_MAX_CACHE_ENTRIES = 100  # Limit to prevent memory leaks


class OllamaModel(BaseModel):
    """Ollama model information"""

    name: str
    description: str | None = None
    pulls: int = 0
    tags: list[str] = []
    updated: str | None = None
    sizes: list[str] = []  # Available parameter sizes like "7b", "13b"
    capabilities: list[str] = []  # e.g., "Vision", "Tools", "Embedding"
    readme: str | None = None  # README/documentation content from model page


def _get_cached(key: str) -> list | None:
    """Get cached value if not expired"""
    if key in _models_cache:
        timestamp, data = _models_cache[key]
        if time.time() - timestamp < CACHE_TTL:
            return data
        del _models_cache[key]
    return None


def _set_cache(key: str, data: list):
    """Set cache value with automatic cleanup of old entries."""
    current_time = time.time()

    # Cleanup expired entries if cache is getting large
    if len(_models_cache) >= _MAX_CACHE_ENTRIES:
        expired_keys = [
            k for k, (ts, _) in _models_cache.items() if current_time - ts >= CACHE_TTL
        ]
        for k in expired_keys:
            _models_cache.pop(k, None)

        # If still too large, remove oldest entries
        if len(_models_cache) >= _MAX_CACHE_ENTRIES:
            sorted_keys = sorted(
                _models_cache.keys(), key=lambda k: _models_cache[k][0]
            )
            for k in sorted_keys[: _MAX_CACHE_ENTRIES // 4]:
                _models_cache.pop(k, None)

    _models_cache[key] = (current_time, data)


def _parse_pulls(text: str) -> int:
    """Parse pull count from text like '100K', '1.5M', '500'"""
    if not text:
        return 0
    text = text.strip().upper()
    try:
        if "M" in text:
            return int(float(text.replace("M", "").replace(",", "")) * 1_000_000)
        elif "K" in text:
            return int(float(text.replace("K", "").replace(",", "")) * 1_000)
        else:
            return int(text.replace(",", ""))
    except ValueError:
        return 0


def _extract_readme(html: str) -> str | None:
    """Extract README content from Ollama model page HTML"""
    import html as html_module

    # The README content is rendered in <div id="display"> on Ollama pages
    # This contains the already-rendered HTML from the markdown
    display_match = re.search(
        r'<div[^>]*id="display"[^>]*>(.*?)</div>\s*</div>\s*<div[^>]*id="editorContainer"',
        html,
        re.DOTALL,
    )
    if display_match:
        content = display_match.group(1).strip()
        if content:
            # Unescape HTML entities
            content = html_module.unescape(content)
            return content

    # Fallback: Try to find the display div with a simpler pattern
    display_match2 = re.search(
        r'<div[^>]*id="display"[^>]*>(.*?)</div>', html, re.DOTALL
    )
    if display_match2:
        content = display_match2.group(1).strip()
        if content:
            content = html_module.unescape(content)
            return content

    return None


def _parse_model_html(html: str) -> list[OllamaModel]:
    """Parse models from ollama.com/library HTML"""
    models = []

    # Simple extraction - get model names from links
    simple_pattern = re.compile(r'href="/library/([a-z0-9_-]+)"', re.IGNORECASE)

    # Find all model names
    model_names = list(set(simple_pattern.findall(html)))

    # Try to extract more info for each model
    for name in model_names:
        # Skip non-model links
        if name in ["", "search", "models", "download"]:
            continue

        model = OllamaModel(name=name)

        # Try to find description and pulls for this model
        # Look for context around the model name
        context_pattern = re.compile(
            rf'href="/library/{re.escape(name)}"[^>]*>.*?'
            rf"(?:<span[^>]*>([^<]+)</span>)?.*?"
            rf"(?:(\d+(?:\.\d+)?[KMB]?)\s*(?:Pulls)?)?",
            re.DOTALL | re.IGNORECASE,
        )
        match = context_pattern.search(html)
        if match:
            if match.group(1):
                model.description = match.group(1).strip()
            if match.group(2):
                model.pulls = _parse_pulls(match.group(2))

        # Check for capability tags
        for cap in ["Vision", "Tools", "Embedding", "Thinking", "Code"]:
            if f">{cap}<" in html or f'"{cap}"' in html.lower():
                # Check if this capability is near this model's entry
                cap_check = re.search(
                    rf'href="/library/{re.escape(name)}".*?{cap}',
                    (
                        html[: html.find(f'href="/library/{name}"') + 2000]
                        if f'href="/library/{name}"' in html
                        else ""
                    ),
                    re.DOTALL | re.IGNORECASE,
                )
                if cap_check:
                    model.capabilities.append(cap.lower())

        models.append(model)

    return models


async def _scrape_ollama_library() -> list[OllamaModel]:
    """Scrape ollama.com/library to get list of models"""
    cached = _get_cached("library")
    if cached:
        return [OllamaModel(**m) if isinstance(m, dict) else m for m in cached]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(OLLAMA_LIBRARY_URL, follow_redirects=True)
            response.raise_for_status()

            models = _parse_model_html(response.text)

            # Cache the results
            _set_cache("library", [m.model_dump() for m in models])
            return models

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, detail=f"Failed to fetch Ollama library: {str(e)}"
        )


# Pre-defined popular models with accurate info
POPULAR_MODELS = [
    OllamaModel(
        name="llama3.3",
        description="New Llama 3.3 70B from Meta with improved reasoning",
        pulls=500000,
        sizes=["70b"],
        capabilities=["tools"],
    ),
    OllamaModel(
        name="llama3.2",
        description="Meta's Llama 3.2 with vision and lightweight models",
        pulls=5000000,
        sizes=["1b", "3b", "11b", "90b"],
        capabilities=["vision"],
    ),
    OllamaModel(
        name="llama3.1",
        description="Meta's flagship model for general tasks",
        pulls=10000000,
        sizes=["8b", "70b", "405b"],
        capabilities=["tools"],
    ),
    OllamaModel(
        name="deepseek-r1",
        description="DeepSeek R1 reasoning model with o1-level performance",
        pulls=2000000,
        sizes=["1.5b", "7b", "8b", "14b", "32b", "70b", "671b"],
        capabilities=["thinking"],
    ),
    OllamaModel(
        name="qwen2.5",
        description="Alibaba's latest Qwen 2.5 series with extended context",
        pulls=3000000,
        sizes=["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"],
        capabilities=["tools"],
    ),
    OllamaModel(
        name="qwen2.5-coder",
        description="Qwen 2.5 optimized for code generation",
        pulls=1500000,
        sizes=["0.5b", "1.5b", "3b", "7b", "14b", "32b"],
        capabilities=["code", "tools"],
    ),
    OllamaModel(
        name="qwq",
        description="Alibaba QwQ reasoning model for complex problems",
        pulls=800000,
        sizes=["32b"],
        capabilities=["thinking"],
    ),
    OllamaModel(
        name="phi4",
        description="Microsoft's Phi-4 14B with strong reasoning",
        pulls=600000,
        sizes=["14b"],
        capabilities=["tools"],
    ),
    OllamaModel(
        name="mistral",
        description="Mistral AI 7B v0.3 - fast and efficient",
        pulls=8000000,
        sizes=["7b"],
        capabilities=["tools"],
    ),
    OllamaModel(
        name="mixtral",
        description="Mistral's MoE model with expert routing",
        pulls=2000000,
        sizes=["8x7b", "8x22b"],
        capabilities=["tools"],
    ),
    OllamaModel(
        name="gemma2",
        description="Google's Gemma 2 with improved performance",
        pulls=2500000,
        sizes=["2b", "9b", "27b"],
        capabilities=["tools"],
    ),
    OllamaModel(
        name="codellama",
        description="Meta's code-specialized Llama model",
        pulls=4000000,
        sizes=["7b", "13b", "34b", "70b"],
        capabilities=["code"],
    ),
    OllamaModel(
        name="llava",
        description="Vision-language model for image understanding",
        pulls=3000000,
        sizes=["7b", "13b", "34b"],
        capabilities=["vision"],
    ),
    OllamaModel(
        name="nomic-embed-text",
        description="High-performance text embedding model",
        pulls=1000000,
        sizes=["137m"],
        capabilities=["embedding"],
    ),
    OllamaModel(
        name="mxbai-embed-large",
        description="Mixedbread.ai embedding model",
        pulls=800000,
        sizes=["335m"],
        capabilities=["embedding"],
    ),
    OllamaModel(
        name="starcoder2",
        description="BigCode StarCoder 2 for code generation",
        pulls=500000,
        sizes=["3b", "7b", "15b"],
        capabilities=["code"],
    ),
    OllamaModel(
        name="yi",
        description="01.AI Yi series models",
        pulls=600000,
        sizes=["6b", "9b", "34b"],
        capabilities=["tools"],
    ),
    OllamaModel(
        name="command-r",
        description="Cohere's Command R for RAG and tools",
        pulls=400000,
        sizes=["35b", "104b"],
        capabilities=["tools"],
    ),
    OllamaModel(
        name="dolphin-mixtral",
        description="Uncensored Mixtral fine-tune by Cognitive",
        pulls=700000,
        sizes=["8x7b", "8x22b"],
        capabilities=["tools"],
    ),
    OllamaModel(
        name="neural-chat",
        description="Intel's neural chat fine-tuned model",
        pulls=300000,
        sizes=["7b"],
        capabilities=[],
    ),
]


@router.get("/models", response_model=list[OllamaModel])
async def list_models(
    search: str | None = Query(None, description="Search query"),
    capability: str | None = Query(
        None,
        description="Filter by capability: vision, tools, embedding, code, thinking",
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
):
    """
    List available Ollama models.
    Uses pre-defined popular models list for reliability.
    """
    models = POPULAR_MODELS.copy()

    # Filter by search query
    if search:
        search_lower = search.lower()
        models = [
            m
            for m in models
            if search_lower in m.name.lower()
            or (m.description and search_lower in m.description.lower())
        ]

    # Filter by capability
    if capability:
        cap_lower = capability.lower()
        models = [m for m in models if cap_lower in m.capabilities]

    # Sort by pulls (popularity)
    models.sort(key=lambda m: m.pulls, reverse=True)

    return models[:limit]


@router.get("/model/{model_name}")
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific Ollama model.
    Fetches data from ollama.com/library/{model_name}.
    """
    # Check cache first
    cache_key = f"model_info_{model_name}"
    cached = _get_cached(cache_key)
    if cached:
        return OllamaModel(**cached[0]) if cached else None

    # Try to scrape info from ollama.com
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{OLLAMA_LIBRARY_URL}/{model_name}",
                follow_redirects=True,
            )

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Model not found")

            response.raise_for_status()
            html = response.text

            # Start with popular model info if available
            model = None
            for popular in POPULAR_MODELS:
                if popular.name == model_name:
                    model = OllamaModel(**popular.model_dump())
                    break

            if not model:
                model = OllamaModel(name=model_name)

            # Extract description if not set
            if not model.description:
                desc_match = re.search(
                    r'<p[^>]*class="[^"]*text-neutral-[^"]*"[^>]*>([^<]+)</p>', html
                )
                if desc_match:
                    model.description = desc_match.group(1).strip()

            # Extract available tags/sizes
            tag_pattern = re.compile(
                r'href="/library/' + re.escape(model_name) + r':([^"]+)"'
            )
            tags = tag_pattern.findall(html)
            if tags:
                model.tags = list(set(tags))

            # Extract parameter sizes from tags
            for tag in model.tags:
                size_match = re.match(r"(\d+(?:\.\d+)?[bm])", tag.lower())
                if size_match and size_match.group(1) not in model.sizes:
                    model.sizes.append(size_match.group(1))

            # Extract README content
            model.readme = _extract_readme(html)

            # Cache the result
            _set_cache(cache_key, [model.model_dump()])

            return model

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Model not found")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, detail=f"Failed to fetch model info: {str(e)}"
        )


@router.get("/tags/{model_name}")
async def get_model_tags(model_name: str):
    """
    Get available tags (versions) for a specific Ollama model.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{OLLAMA_LIBRARY_URL}/{model_name}/tags",
                follow_redirects=True,
            )

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Model not found")

            response.raise_for_status()
            html = response.text

            # Extract tags from the page
            tag_pattern = re.compile(
                rf'href="/library/{re.escape(model_name)}:([^"]+)"', re.IGNORECASE
            )
            tags = list(set(tag_pattern.findall(html)))

            # Parse tag info
            tag_info = []
            for tag in tags:
                info = {"name": tag, "full_name": f"{model_name}:{tag}"}

                # Try to determine size from tag name
                size_match = re.match(r"(\d+(?:\.\d+)?[bm])", tag.lower())
                if size_match:
                    info["size"] = size_match.group(1)

                # Check for quantization
                for quant in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "fp16", "fp32"]:
                    if quant in tag.lower():
                        info["quantization"] = quant
                        break

                tag_info.append(info)

            return {"model": model_name, "tags": tag_info}

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Model not found")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, detail=f"Failed to fetch model tags: {str(e)}"
        )


@router.get("/popular", response_model=list[OllamaModel])
async def get_popular_models(
    limit: int = Query(20, ge=1, le=50, description="Number of results"),
):
    """
    Get popular Ollama models sorted by downloads.
    """
    models = POPULAR_MODELS.copy()
    models.sort(key=lambda m: m.pulls, reverse=True)
    return models[:limit]

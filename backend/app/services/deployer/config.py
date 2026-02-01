"""Backend configuration builders for deployment.

This module contains functions that build deployment configurations
for different inference backends (vLLM, SGLang, Ollama).
"""

import logging
from typing import TYPE_CHECKING

from app.config import get_settings
from app.models.llm_model import BackendType

if TYPE_CHECKING:
    from app.models.deployment import Deployment
    from app.models.llm_model import LLMModel

logger = logging.getLogger(__name__)
settings = get_settings()


def build_deploy_request(deployment: "Deployment") -> dict:
    """Build the deployment request for worker agent.

    Supports multiple backends:
    - vLLM: High-throughput inference with OpenAI-compatible API
    - Ollama: Simple local LLM inference with OpenAI-compatible API
    """
    model = deployment.model

    # Determine docker image based on backend
    # Priority: deployment extra_params > model docker_image > backend default
    deployment_image = (
        deployment.extra_params.get("docker_image") if deployment.extra_params else None
    )

    backend = deployment.backend

    if deployment_image:
        image = deployment_image
    elif model.docker_image:
        image = model.docker_image
    elif backend == BackendType.VLLM.value:
        image = settings.vllm_default_image
    elif backend == BackendType.SGLANG.value:
        image = settings.sglang_default_image
    elif backend == BackendType.OLLAMA.value:
        image = settings.ollama_default_image
    else:
        logger.warning(f"Unknown backend: {backend}, defaulting to vLLM")
        image = settings.vllm_default_image

    # Build command based on backend type
    if backend == BackendType.OLLAMA.value:
        cmd, env = build_ollama_config(model, deployment)
    elif backend == BackendType.SGLANG.value:
        cmd, env = build_sglang_config(model, deployment)
    else:
        cmd, env = build_vllm_config(model, deployment)

    request = {
        "deployment_id": deployment.id,
        "deployment_name": deployment.name,
        "image": image,
        "command": cmd,
        "model_id": model.model_id,
        "gpu_indexes": deployment.gpu_indexes or [0],
        "environment": env,
    }

    # Note: We don't reuse existing port to avoid conflicts.
    # Worker will automatically allocate an available port.

    return request


def build_vllm_config(
    model: "LLMModel",
    deployment: "Deployment",
) -> tuple[list[str], dict[str, str]]:
    """Build vLLM container command and environment."""
    cmd = [
        "--model",
        model.model_id,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]

    # Add default params if any
    if model.default_params:
        for key, value in model.default_params.items():
            if value is True:
                cmd.append(f"--{key}")
            elif value is not False and value is not None:
                cmd.extend([f"--{key}", str(value)])

    # Add extra params if any (skip special keys like docker_image, custom_args)
    if deployment.extra_params:
        skip_keys = {"docker_image", "custom_args"}
        for key, value in deployment.extra_params.items():
            if key in skip_keys:
                continue
            if value is True:
                cmd.append(f"--{key}")
                # Auto-add tool-call-parser when enable-auto-tool-choice is enabled
                if key == "enable-auto-tool-choice":
                    cmd.extend(["--tool-call-parser", "hermes"])
            elif value is not False and value is not None:
                cmd.extend([f"--{key}", str(value)])

        # Handle custom CLI arguments
        custom_args = deployment.extra_params.get("custom_args")
        if custom_args and isinstance(custom_args, str):
            # Parse custom args: split by newlines and spaces
            for line in custom_args.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    # Split each line by spaces for multi-arg support
                    cmd.extend(line.split())

    env = {
        "HF_HOME": "/root/.cache/huggingface",
    }

    return cmd, env


def build_sglang_config(
    model: "LLMModel",
    deployment: "Deployment",
) -> tuple[list[str], dict[str, str]]:
    """Build SGLang container command and environment.

    SGLang uses similar command-line arguments to vLLM but with some
    differences in parameter names. Unlike vLLM, the sglang Docker image
    does not have a proper ENTRYPOINT, so we need to explicitly specify
    the launch command.
    """
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model.model_id,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]

    # Add default params if any
    if model.default_params:
        for key, value in model.default_params.items():
            if value is True:
                cmd.append(f"--{key}")
            elif value is not False and value is not None:
                cmd.extend([f"--{key}", str(value)])

    # Add extra params if any (skip special keys like docker_image, custom_args)
    if deployment.extra_params:
        skip_keys = {"docker_image", "custom_args"}
        for key, value in deployment.extra_params.items():
            if key in skip_keys:
                continue
            if value is True:
                cmd.append(f"--{key}")
            elif value is not False and value is not None:
                cmd.extend([f"--{key}", str(value)])

        # Handle custom CLI arguments
        custom_args = deployment.extra_params.get("custom_args")
        if custom_args and isinstance(custom_args, str):
            # Parse custom args: split by newlines and spaces
            for line in custom_args.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    # Split each line by spaces for multi-arg support
                    cmd.extend(line.split())

    env = {
        "HF_HOME": "/root/.cache/huggingface",
    }

    return cmd, env


def build_ollama_config(
    model: "LLMModel",
    deployment: "Deployment",
) -> tuple[list[str], dict[str, str]]:
    """Build Ollama container command and environment.

    Ollama uses environment variables for configuration instead of
    command-line arguments. The model is pulled and run via API after
    the container starts.
    """
    # Ollama's default entrypoint is "ollama serve"
    cmd = ["serve"]

    # Ollama environment configuration
    env = {
        "OLLAMA_HOST": "0.0.0.0:8000",  # Bind to container port 8000
        "OLLAMA_ORIGINS": "*",  # Allow CORS from all origins (required for web UI)
        "OLLAMA_NUM_PARALLEL": str(
            deployment.extra_params.get("num_parallel", 4) if deployment.extra_params else "4"
        ),
        "OLLAMA_MAX_LOADED_MODELS": str(
            deployment.extra_params.get("max_loaded_models", 1) if deployment.extra_params else "1"
        ),
        # GPU settings
        "OLLAMA_GPU_OVERHEAD": "0",
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in (deployment.gpu_indexes or [0])),
    }

    # Add any custom environment variables from extra_params
    if deployment.extra_params:
        for key, value in deployment.extra_params.items():
            if key.startswith("OLLAMA_") and value is not None:
                env[key] = str(value)

        # Handle custom environment variables from custom_args
        custom_args = deployment.extra_params.get("custom_args")
        if custom_args and isinstance(custom_args, str):
            # Parse custom args as environment variables (KEY=VALUE format)
            for line in custom_args.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip()

    return cmd, env

"""Utility functions for the tuning agent.

This module contains helper functions used across the tuning package.
"""


def extract_model_family(model_name: str) -> str:
    """Extract model family from name.

    Args:
        model_name: The model name or ID (e.g., "Qwen/Qwen3-0.6B")

    Returns:
        The model family name (e.g., "Qwen") or "Unknown" if not recognized
    """
    name_lower = model_name.lower()
    families = {
        "qwen": "Qwen",
        "llama": "Llama",
        "mistral": "Mistral",
        "deepseek": "DeepSeek",
        "phi": "Phi",
        "gemma": "Gemma",
        "yi": "Yi",
        "glm": "GLM",
    }
    for key, value in families.items():
        if key in name_lower:
            return value
    return "Unknown"

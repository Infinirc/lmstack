"""
Benchmark Configuration

Defines configuration options for benchmark execution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LoadPattern(Enum):
    """Load pattern types for benchmark execution"""

    # Fixed concurrency throughout the test
    FIXED = "fixed"

    # Gradually increase concurrency to find saturation point
    INCREMENTAL = "incremental"

    # Burst load pattern with spikes
    BURST = "burst"

    # Step-wise increase for finding optimal concurrency
    STEP = "step"


@dataclass
class SaturationConfig:
    """Configuration for saturation detection"""

    # Enable saturation detection (auto-find optimal concurrency)
    enabled: bool = False

    # Starting concurrency level
    start_concurrency: int = 1

    # Maximum concurrency to test
    max_concurrency: int = 64

    # Concurrency step size
    step_size: int = 2

    # Multiplier for exponential stepping (alternative to linear)
    step_multiplier: float = 1.5

    # Use exponential stepping instead of linear
    use_exponential: bool = True

    # Minimum requests per concurrency level
    requests_per_level: int = 20

    # Throughput degradation threshold (e.g., 0.95 = 5% drop triggers saturation)
    degradation_threshold: float = 0.95

    # Latency increase threshold (e.g., 1.5 = 50% increase triggers saturation)
    latency_threshold: float = 1.5

    # Number of consecutive degradations before declaring saturation
    consecutive_degradations: int = 2


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""

    # Target endpoint (OpenAI-compatible API)
    endpoint: str

    # Model name/ID for API requests
    model_name: str

    # Load pattern to use
    load_pattern: LoadPattern = LoadPattern.FIXED

    # Concurrency level (for FIXED pattern)
    concurrency: int = 10

    # Duration in seconds (0 = use num_requests instead)
    duration_seconds: int = 0

    # Number of requests (used when duration_seconds = 0)
    num_requests: int = 50

    # Warmup requests before actual benchmark
    warmup_requests: int = 5

    # Request timeout in seconds
    request_timeout: float = 120.0

    # Prompt configuration
    prompt_tokens: int = 256  # Approximate input tokens
    output_tokens: int = 128  # Max output tokens

    # Custom prompt (overrides prompt_tokens if set)
    custom_prompt: str | None = None

    # Saturation detection config
    saturation: SaturationConfig = field(default_factory=SaturationConfig)

    # Extra parameters for requests
    extra_params: dict[str, Any] = field(default_factory=dict)

    # Stream responses (required for accurate ITL measurement)
    stream: bool = True

    # Verbose logging
    verbose: bool = False

    def get_prompt(self) -> str:
        """Get the prompt to use for benchmarking"""
        if self.custom_prompt:
            return self.custom_prompt

        # Generate synthetic prompt of approximately prompt_tokens length
        # Average English word is ~4 characters, ~1.3 tokens
        words_needed = int(self.prompt_tokens / 1.3)
        base_prompt = "Explain the following concept in detail: "
        filler = "artificial intelligence machine learning neural network deep learning natural language processing computer vision reinforcement learning transformer architecture attention mechanism gradient descent backpropagation optimization algorithm data science statistical analysis pattern recognition feature extraction dimensionality reduction clustering classification regression prediction inference training validation testing deployment scalability performance efficiency accuracy precision recall "

        # Repeat filler to reach desired length
        repeated = (filler * ((words_needed // len(filler.split())) + 1)).split()
        prompt = base_prompt + " ".join(repeated[:words_needed])

        return prompt

    def validate(self) -> tuple[bool, str]:
        """Validate configuration"""
        if not self.endpoint:
            return False, "Endpoint is required"

        if not self.model_name:
            return False, "Model name is required"

        if self.concurrency < 1:
            return False, "Concurrency must be at least 1"

        if self.duration_seconds < 0:
            return False, "Duration cannot be negative"

        if self.num_requests < 1 and self.duration_seconds == 0:
            return False, "Either duration_seconds or num_requests must be positive"

        if self.request_timeout < 1:
            return False, "Request timeout must be at least 1 second"

        return True, ""

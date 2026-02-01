"""
Benchmark Runner

Executes benchmarks against OpenAI-compatible endpoints with accurate
token-level timing measurements using streaming responses.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import BenchmarkConfig, LoadPattern
from .metrics import BenchmarkMetrics, RequestResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Complete benchmark result"""

    metrics: BenchmarkMetrics
    config: BenchmarkConfig
    raw_results: list[RequestResult] = field(default_factory=list)
    error: str | None = None
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Total benchmark duration"""
        return self.completed_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metrics": self.metrics.to_dict(),
            "config": {
                "endpoint": self.config.endpoint,
                "model_name": self.config.model_name,
                "load_pattern": self.config.load_pattern.value,
                "concurrency": self.config.concurrency,
                "num_requests": self.config.num_requests,
                "duration_seconds": self.config.duration_seconds,
            },
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
        }


class BenchmarkRunner:
    """
    Executes benchmarks against LLM inference endpoints.

    Supports multiple load patterns and accurate token-level timing.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._cancelled = False
        self._results: list[RequestResult] = []
        self._active_requests = 0
        self._completed_requests = 0
        self._start_time: float = 0.0

    async def run(self) -> BenchmarkResult:
        """Execute the benchmark"""
        # Validate config
        valid, error = self.config.validate()
        if not valid:
            return BenchmarkResult(
                metrics=BenchmarkMetrics(),
                config=self.config,
                error=error,
            )

        self._start_time = time.time()
        result = BenchmarkResult(
            metrics=BenchmarkMetrics(),
            config=self.config,
            started_at=self._start_time,
        )

        try:
            # Run warmup
            if self.config.warmup_requests > 0:
                await self._run_warmup()

            # Execute based on load pattern
            if self.config.load_pattern == LoadPattern.FIXED:
                await self._run_fixed_load()
            elif self.config.load_pattern == LoadPattern.INCREMENTAL:
                await self._run_incremental_load()
            elif self.config.load_pattern == LoadPattern.BURST:
                await self._run_burst_load()
            elif self.config.load_pattern == LoadPattern.STEP:
                await self._run_step_load()
            else:
                await self._run_fixed_load()

            # Calculate metrics
            end_time = time.time()
            total_duration = end_time - self._start_time

            result.metrics = BenchmarkMetrics.from_results(
                self._results,
                total_duration,
                self.config.concurrency,
            )
            result.raw_results = self._results
            result.completed_at = end_time

        except Exception as e:
            logger.exception(f"Benchmark failed: {e}")
            result.error = str(e)
            result.completed_at = time.time()

        return result

    def cancel(self):
        """Cancel the running benchmark"""
        self._cancelled = True

    async def _run_warmup(self):
        """Run warmup requests"""
        if self.config.verbose:
            logger.info(f"Running {self.config.warmup_requests} warmup requests...")

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            tasks = [
                self._make_request(client, warmup=True) for _ in range(self.config.warmup_requests)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        if self.config.verbose:
            logger.info("Warmup complete")

    async def _run_fixed_load(self):
        """Run with fixed concurrency"""
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def limited_request(client: httpx.AsyncClient):
            async with semaphore:
                return await self._make_request(client)

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            if self.config.duration_seconds > 0:
                # Duration-based
                end_time = time.time() + self.config.duration_seconds
                tasks: list[asyncio.Task] = []

                while time.time() < end_time and not self._cancelled:
                    if len(tasks) < self.config.concurrency * 2:
                        task = asyncio.create_task(limited_request(client))
                        tasks.append(task)

                    # Clean up completed tasks
                    done = [t for t in tasks if t.done()]
                    for t in done:
                        try:
                            result = t.result()
                            if result:
                                self._results.append(result)
                        except Exception:
                            pass
                        tasks.remove(t)

                    await asyncio.sleep(0.01)

                # Wait for remaining tasks
                for task in tasks:
                    try:
                        result = await task
                        if result:
                            self._results.append(result)
                    except Exception:
                        pass
            else:
                # Request count-based
                tasks = [limited_request(client) for _ in range(self.config.num_requests)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in results:
                    if isinstance(r, RequestResult):
                        self._results.append(r)

    async def _run_incremental_load(self):
        """Run with incrementally increasing concurrency"""
        sat_config = self.config.saturation
        current_concurrency = sat_config.start_concurrency

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            while current_concurrency <= sat_config.max_concurrency and not self._cancelled:
                if self.config.verbose:
                    logger.info(f"Testing concurrency level: {current_concurrency}")

                semaphore = asyncio.Semaphore(current_concurrency)

                async def limited_request(sem=semaphore):
                    async with sem:
                        return await self._make_request(client)

                tasks = [limited_request() for _ in range(sat_config.requests_per_level)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in results:
                    if isinstance(r, RequestResult):
                        self._results.append(r)

                # Increase concurrency
                if sat_config.use_exponential:
                    current_concurrency = int(current_concurrency * sat_config.step_multiplier)
                else:
                    current_concurrency += sat_config.step_size

    async def _run_burst_load(self):
        """Run with burst load pattern"""
        # Burst pattern: low -> high -> low
        concurrency_levels = [
            self.config.concurrency // 4,  # Low
            self.config.concurrency,  # High (burst)
            self.config.concurrency // 4,  # Low
            self.config.concurrency,  # High (burst)
            self.config.concurrency // 4,  # Low
        ]

        requests_per_phase = self.config.num_requests // len(concurrency_levels)

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            for phase, concurrency in enumerate(concurrency_levels):
                if self._cancelled:
                    break

                if self.config.verbose:
                    logger.info(f"Burst phase {phase + 1}: concurrency={concurrency}")

                semaphore = asyncio.Semaphore(max(1, concurrency))

                async def limited_request(sem=semaphore):
                    async with sem:
                        return await self._make_request(client)

                tasks = [limited_request() for _ in range(requests_per_phase)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in results:
                    if isinstance(r, RequestResult):
                        self._results.append(r)

    async def _run_step_load(self):
        """Run with step-wise load increase"""
        sat_config = self.config.saturation
        steps = []

        # Generate step levels
        current = sat_config.start_concurrency
        while current <= sat_config.max_concurrency:
            steps.append(current)
            if sat_config.use_exponential:
                current = int(current * sat_config.step_multiplier)
            else:
                current += sat_config.step_size

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            for step_idx, concurrency in enumerate(steps):
                if self._cancelled:
                    break

                if self.config.verbose:
                    logger.info(f"Step {step_idx + 1}/{len(steps)}: concurrency={concurrency}")

                semaphore = asyncio.Semaphore(concurrency)

                async def limited_request(sem=semaphore):
                    async with sem:
                        return await self._make_request(client)

                tasks = [limited_request() for _ in range(sat_config.requests_per_level)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in results:
                    if isinstance(r, RequestResult):
                        self._results.append(r)

    async def _make_request(
        self,
        client: httpx.AsyncClient,
        warmup: bool = False,
    ) -> RequestResult | None:
        """Make a single request with token-level timing"""
        if self._cancelled:
            return None

        result = RequestResult(start_time=time.time())
        prompt = self.config.get_prompt()

        request_body = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.output_tokens,
            "stream": self.config.stream,
            **self.config.extra_params,
        }

        endpoint = self.config.endpoint.rstrip("/")
        url = f"{endpoint}/chat/completions"

        try:
            start_time = time.perf_counter()

            if self.config.stream:
                result = await self._make_streaming_request(client, url, request_body, start_time)
            else:
                result = await self._make_non_streaming_request(
                    client, url, request_body, start_time
                )

            result.end_time = time.time()

            if self.config.verbose and not warmup:
                logger.info(
                    f"Request completed: TTFT={result.ttft_ms:.1f}ms, "
                    f"tokens={result.completion_tokens}, "
                    f"latency={result.total_latency_ms:.1f}ms"
                )

        except httpx.TimeoutException:
            result.success = False
            result.error_message = "Request timeout"
            result.end_time = time.time()

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.end_time = time.time()

        return result

    async def _make_streaming_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        request_body: dict,
        start_time: float,
    ) -> RequestResult:
        """Make a streaming request with token-level timing"""
        result = RequestResult(start_time=time.time())
        first_token_received = False
        token_timestamps: list[float] = []
        token_count = 0

        async with client.stream("POST", url, json=request_body) as response:
            if response.status_code != 200:
                result.success = False
                result.error_message = f"HTTP {response.status_code}"
                return result

            async for line in response.aiter_lines():
                if self._cancelled:
                    break

                if not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        current_time = time.perf_counter()

                        if not first_token_received:
                            result.ttft_ms = (current_time - start_time) * 1000
                            first_token_received = True

                        # Record timestamp for each token
                        token_timestamps.append((current_time - start_time) * 1000)
                        token_count += 1

                    # Extract usage if available (some backends include it)
                    usage = chunk.get("usage", {})
                    if usage:
                        result.prompt_tokens = usage.get("prompt_tokens", result.prompt_tokens)
                        result.completion_tokens = usage.get("completion_tokens", token_count)

                except json.JSONDecodeError:
                    continue

        end_time = time.perf_counter()
        result.total_latency_ms = (end_time - start_time) * 1000
        result.token_timestamps_ms = token_timestamps
        result.completion_tokens = (
            token_count if result.completion_tokens == 0 else result.completion_tokens
        )

        return result

    async def _make_non_streaming_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        request_body: dict,
        start_time: float,
    ) -> RequestResult:
        """Make a non-streaming request"""
        result = RequestResult(start_time=time.time())

        response = await client.post(url, json=request_body)
        end_time = time.perf_counter()

        if response.status_code != 200:
            result.success = False
            result.error_message = f"HTTP {response.status_code}"
            return result

        try:
            data = response.json()
            usage = data.get("usage", {})
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)

            # For non-streaming, TTFT = total latency
            result.total_latency_ms = (end_time - start_time) * 1000
            result.ttft_ms = result.total_latency_ms

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result


async def run_benchmark(
    endpoint: str,
    model_name: str,
    concurrency: int = 10,
    num_requests: int = 50,
    warmup_requests: int = 5,
    output_tokens: int = 128,
    verbose: bool = False,
) -> BenchmarkResult:
    """
    Convenience function to run a simple benchmark.

    Args:
        endpoint: OpenAI-compatible API endpoint
        model_name: Model name/ID
        concurrency: Number of concurrent requests
        num_requests: Total number of requests
        warmup_requests: Number of warmup requests
        output_tokens: Max output tokens per request
        verbose: Enable verbose logging

    Returns:
        BenchmarkResult with metrics
    """
    config = BenchmarkConfig(
        endpoint=endpoint,
        model_name=model_name,
        concurrency=concurrency,
        num_requests=num_requests,
        warmup_requests=warmup_requests,
        output_tokens=output_tokens,
        verbose=verbose,
    )

    runner = BenchmarkRunner(config)
    return await runner.run()

"""
Bayesian Optimization-based Auto-Tuning Service

Uses Optuna's TPE (Tree-structured Parzen Estimator) for efficient
hyperparameter search. This replaces the LLM Agent approach with
systematic Bayesian optimization while maintaining MCP-compatible
tool interfaces.

Key concepts:
- Bayesian Optimization: Uses surrogate model + acquisition function
- Filter-Scorer Architecture: Filters invalid configs, scores valid ones
- Knowledge Transfer: Uses historical results to warm-start optimization
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import httpx
import optuna
from optuna.samplers import TPESampler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import async_session_maker
from app.models.deployment import Deployment, DeploymentStatus
from app.models.tuning import OptimizationTarget, PerformanceKnowledge, TuningJob, TuningJobStatus
from app.models.worker import Worker
from app.services.deployer import DeployerService

# Configure logging with detailed format
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SearchSpace(Enum):
    """Parameter search space types"""

    CATEGORICAL = "categorical"
    INTEGER = "integer"
    FLOAT = "float"
    LOG_FLOAT = "log_float"


@dataclass
class TuningParameter:
    """Definition of a tunable parameter"""

    name: str
    space_type: SearchSpace
    choices: list[Any] | None = None  # For categorical
    low: float | None = None  # For numeric
    high: float | None = None  # For numeric
    step: float | None = None  # For discrete numeric
    default: Any = None

    def suggest(self, trial: optuna.Trial) -> Any:
        """Generate parameter suggestion from Optuna trial"""
        if self.space_type == SearchSpace.CATEGORICAL:
            return trial.suggest_categorical(self.name, self.choices)
        elif self.space_type == SearchSpace.INTEGER:
            return trial.suggest_int(
                self.name, int(self.low), int(self.high), step=int(self.step or 1)
            )
        elif self.space_type == SearchSpace.FLOAT:
            return trial.suggest_float(self.name, self.low, self.high, step=self.step)
        elif self.space_type == SearchSpace.LOG_FLOAT:
            return trial.suggest_float(self.name, self.low, self.high, log=True)
        return self.default


@dataclass
class HardwareProfile:
    """Hardware characteristics for filtering"""

    gpu_name: str
    gpu_count: int
    vram_per_gpu_gb: float
    total_vram_gb: float
    compute_capability: str = "unknown"

    @classmethod
    def from_worker(cls, worker: Worker) -> "HardwareProfile":
        """Build profile from Worker model"""
        gpu_info = worker.gpu_info or []
        if not gpu_info:
            return cls(
                gpu_name="Unknown",
                gpu_count=0,
                vram_per_gpu_gb=0,
                total_vram_gb=0,
            )

        # Normalize memory values to GB
        def normalize_memory(mem: int | float) -> float:
            if mem > 1_000_000_000:
                return mem / (1024**3)
            elif mem > 1_000_000:
                return mem / (1024**2)
            elif mem > 1000:
                return mem / 1024
            return float(mem)

        first_gpu = gpu_info[0]
        vram = normalize_memory(first_gpu.get("memory_total", 0))

        return cls(
            gpu_name=first_gpu.get("name", "Unknown"),
            gpu_count=len(gpu_info),
            vram_per_gpu_gb=vram,
            total_vram_gb=vram * len(gpu_info),
        )


@dataclass
class TrialOutcome:
    """Result from a single trial execution"""

    trial_id: int
    parameters: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: str | None = None
    duration_seconds: float = 0.0
    deployment_id: int | None = None


def build_llm_search_space(hardware: HardwareProfile) -> list[TuningParameter]:
    """
    Construct parameter search space based on hardware capabilities.

    This implements the Filter phase of Filter-Scorer architecture:
    - Filters out invalid parameter combinations
    - Adapts ranges based on GPU capabilities
    """
    params = []

    # Engine selection (always include vLLM, conditionally add others)
    engines = ["vllm"]
    if hardware.vram_per_gpu_gb >= 8:  # SGLang needs decent VRAM
        engines.append("sglang")

    params.append(
        TuningParameter(
            name="engine",
            space_type=SearchSpace.CATEGORICAL,
            choices=engines,
            default="vllm",
        )
    )

    # GPU memory utilization (conservative for smaller GPUs)
    if hardware.vram_per_gpu_gb < 12:
        mem_range = (0.7, 0.85)
    elif hardware.vram_per_gpu_gb < 24:
        mem_range = (0.75, 0.92)
    else:
        mem_range = (0.8, 0.95)

    params.append(
        TuningParameter(
            name="gpu_memory_utilization",
            space_type=SearchSpace.FLOAT,
            low=mem_range[0],
            high=mem_range[1],
            step=0.05,
            default=0.85,
        )
    )

    # Max concurrent sequences
    params.append(
        TuningParameter(
            name="max_num_seqs",
            space_type=SearchSpace.INTEGER,
            low=4,
            high=64 if hardware.vram_per_gpu_gb >= 24 else 32,
            step=4,
            default=16,
        )
    )

    # Tensor parallelism (only if multiple GPUs)
    if hardware.gpu_count > 1:
        tp_choices = [1]
        if hardware.gpu_count >= 2:
            tp_choices.append(2)
        if hardware.gpu_count >= 4:
            tp_choices.append(4)
        if hardware.gpu_count >= 8:
            tp_choices.append(8)

        params.append(
            TuningParameter(
                name="tensor_parallel_size",
                space_type=SearchSpace.CATEGORICAL,
                choices=tp_choices,
                default=1,
            )
        )

    return params


class ConfigurationFilter:
    """
    Validates parameter configurations against hardware constraints.

    Implements Filter phase: reject invalid configs before evaluation.
    """

    def __init__(self, hardware: HardwareProfile, model_size_gb: float = 7.0):
        self.hardware = hardware
        self.estimated_model_size = model_size_gb

    def is_valid(self, params: dict[str, Any]) -> tuple[bool, str]:
        """Check if configuration is valid for the hardware"""

        # Check VRAM requirement
        tp_size = params.get("tensor_parallel_size", 1)
        mem_util = params.get("gpu_memory_utilization", 0.9)

        # Rough VRAM estimation: model_size / tp_size + overhead
        estimated_vram = (self.estimated_model_size / tp_size) + 2.0  # 2GB overhead
        available_vram = self.hardware.vram_per_gpu_gb * mem_util

        if estimated_vram > available_vram:
            return (
                False,
                f"Estimated VRAM ({estimated_vram:.1f}GB) exceeds available ({available_vram:.1f}GB)",
            )

        # Check tensor parallelism divisibility
        if tp_size > self.hardware.gpu_count:
            return False, f"TP size {tp_size} exceeds GPU count {self.hardware.gpu_count}"

        return True, ""


class ObjectiveCalculator:
    """
    Computes optimization objective from benchmark metrics.

    Implements Scorer phase: score valid configurations by performance.
    """

    def __init__(self, target: OptimizationTarget):
        self.target = target

    def compute(self, metrics: dict[str, float]) -> float:
        """
        Calculate objective value (higher is better for maximization).

        For Optuna, we negate values when minimizing so all objectives
        are treated as maximization internally.
        """
        throughput = metrics.get("throughput_tps", 0.0)
        ttft = metrics.get("avg_ttft_ms", float("inf"))
        tpot = metrics.get("avg_tpot_ms", float("inf"))

        if self.target == OptimizationTarget.THROUGHPUT:
            # Maximize throughput
            return throughput

        elif self.target == OptimizationTarget.LATENCY:
            # Minimize latency (negate for maximization)
            if ttft == float("inf"):
                return float("-inf")
            return -1.0 * (ttft + tpot * 10)  # Weight TPOT more

        elif self.target == OptimizationTarget.BALANCED:
            # Combined score: throughput / latency
            if ttft == 0 or throughput == 0:
                return float("-inf")
            latency_factor = 1 + (ttft / 100) + (tpot / 10)
            return throughput / latency_factor

        elif self.target == OptimizationTarget.COST:
            # Maximize efficiency (throughput per resource unit)
            # For now, just use throughput as proxy
            return throughput

        return throughput


class BayesianTuningService:
    """
    Main tuning service using Bayesian optimization.

    Workflow:
    1. Analyze hardware and build search space
    2. Query knowledge base for warm-start
    3. Run Optuna optimization loop
    4. Each trial: deploy -> benchmark -> cleanup
    5. Report best configuration
    """

    def __init__(
        self,
        db: AsyncSession,
        job: TuningJob,
        n_trials: int = 10,
        timeout_per_trial: int = 600,
    ):
        self.db = db
        self.job = job
        self.n_trials = n_trials
        self.timeout_per_trial = timeout_per_trial
        self.deployer = DeployerService()

        self._current_deployment_id: int | None = None
        self._cancelled = False
        self._outcomes: list[TrialOutcome] = []
        self._logs: list[dict[str, str]] = []

    async def _log(self, level: str, message: str):
        """Log message to both logger and database"""
        timestamp = datetime.now(UTC).isoformat()
        log_entry = {"timestamp": timestamp, "level": level, "message": message}
        self._logs.append(log_entry)

        # Also log to standard logger
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)

        # Save to database (limit to last 100 logs)
        self.job.logs = self._logs[-100:]
        await self.db.commit()

    async def run(self) -> dict[str, Any]:
        """Execute the tuning process"""
        start_time = time.time()

        await self._log("INFO", "=" * 50)
        await self._log("INFO", "Starting Bayesian Optimization")
        await self._log("INFO", f"Model: {self.job.model.name}")
        await self._log("INFO", f"Target: {self.job.optimization_target}")
        await self._log("INFO", f"Trials: {self.n_trials}")
        await self._log("INFO", "=" * 50)

        try:
            # Phase 1: Hardware analysis
            await self._log("INFO", "Phase 1: Analyzing hardware...")
            await self._update_status(
                TuningJobStatus.ANALYZING, "Analyzing hardware configuration..."
            )
            hardware = await self._analyze_hardware()

            if hardware.gpu_count == 0:
                raise RuntimeError("No GPUs detected on worker")

            await self._log("INFO", f"Hardware: {hardware.gpu_count}x {hardware.gpu_name}")
            await self._log("INFO", f"VRAM: {hardware.total_vram_gb:.1f} GB total")

            # Phase 2: Query knowledge base
            await self._log("INFO", "Phase 2: Querying knowledge base...")
            await self._update_status(
                TuningJobStatus.QUERYING_KB, "Checking historical performance data..."
            )
            warm_start_params = await self._query_knowledge_base(hardware)

            if warm_start_params:
                await self._log("INFO", f"Found {len(warm_start_params)} warm-start configurations")
            else:
                await self._log("INFO", "No historical data found, starting fresh")

            # Phase 3: Build search space
            await self._log("INFO", "Phase 3: Building search space...")
            search_space = build_llm_search_space(hardware)
            config_filter = ConfigurationFilter(hardware, model_size_gb=self._estimate_model_size())
            objective_calc = ObjectiveCalculator(OptimizationTarget(self.job.optimization_target))

            await self._log("INFO", f"Search space: {[p.name for p in search_space]}")

            # Phase 4: Create Optuna study
            await self._log("INFO", "Phase 4: Initializing TPE sampler...")
            sampler = TPESampler(
                seed=42,
                n_startup_trials=min(3, self.n_trials // 2),
                multivariate=True,  # Consider parameter correlations
            )

            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                study_name=f"lmstack_tuning_{self.job.id}",
            )

            # Add warm-start trials from knowledge base
            if warm_start_params:
                for params in warm_start_params[:2]:  # At most 2 warm-start
                    try:
                        study.enqueue_trial(params)
                        await self._log("INFO", f"Enqueued warm-start: {params}")
                    except Exception as e:
                        await self._log("WARNING", f"Failed to enqueue warm-start: {e}")

            # Phase 5: Optimization loop
            await self._log("INFO", "Phase 5: Starting optimization loop...")
            await self._update_status(TuningJobStatus.EXPLORING, "Starting optimization...")

            for trial_idx in range(self.n_trials):
                if self._cancelled:
                    break

                # Check cancellation
                await self.db.refresh(self.job)
                if self.job.status == TuningJobStatus.CANCELLED.value:
                    self._cancelled = True
                    await self._log("INFO", "Cancelled by user")
                    break

                await self._log("INFO", "-" * 40)
                await self._log("INFO", f"Trial {trial_idx + 1}/{self.n_trials}")

                # Generate trial parameters
                trial = study.ask()
                params = {p.name: p.suggest(trial) for p in search_space}
                await self._log("INFO", f"Parameters: {params}")

                # Filter invalid configurations
                is_valid, reason = config_filter.is_valid(params)
                if not is_valid:
                    await self._log("WARNING", f"Skipped: {reason}")
                    study.tell(trial, float("-inf"))
                    continue

                # Update progress
                await self._update_progress(
                    step=trial_idx + 1,
                    total=self.n_trials,
                    step_name=f"Trial {trial_idx + 1}",
                    params=params,
                )

                # Execute trial
                outcome = await self._execute_trial(trial_idx, params)
                self._outcomes.append(outcome)

                # Report to Optuna
                if outcome.success and outcome.metrics:
                    objective = objective_calc.compute(outcome.metrics)
                    study.tell(trial, objective)
                    await self._log("INFO", f"Trial {trial_idx + 1} completed:")
                    await self._log("INFO", f"  Objective: {objective:.2f}")
                    await self._log(
                        "INFO", f"  TPS: {outcome.metrics.get('throughput_tps', 0):.1f}"
                    )
                    await self._log(
                        "INFO", f"  TTFT: {outcome.metrics.get('avg_ttft_ms', 0):.0f} ms"
                    )
                else:
                    study.tell(trial, float("-inf"))
                    await self._log(
                        "ERROR", f"Trial {trial_idx + 1} failed: {outcome.error_message}"
                    )

            # Phase 6: Finalize results
            await self._log("INFO", "=" * 50)
            await self._log("INFO", "Phase 6: Finalizing results...")

            if self._cancelled:
                await self._update_status(TuningJobStatus.CANCELLED, "Tuning was cancelled")
                await self._log("INFO", "Job cancelled")
                return {"success": False, "reason": "cancelled"}

            # Get best result
            best_trial = study.best_trial
            best_params = best_trial.params
            best_value = best_trial.value

            # Find corresponding outcome for full metrics
            best_outcome = None
            for outcome in self._outcomes:
                if outcome.parameters == best_params:
                    best_outcome = outcome
                    break

            # Save to knowledge base
            await self._save_to_knowledge_base(best_params, best_outcome)

            # Update job with results
            self.job.best_config = {
                **best_params,
                "objective_value": best_value,
                "metrics": best_outcome.metrics if best_outcome else {},
            }
            self.job.all_results = [
                {
                    "parameters": o.parameters,
                    "metrics": o.metrics,
                    "success": o.success,
                    "error": o.error_message,
                }
                for o in self._outcomes
            ]

            elapsed = time.time() - start_time
            await self._log("INFO", "=" * 50)
            await self._log("INFO", "Optimization Complete!")
            await self._log("INFO", f"Duration: {elapsed/60:.1f} minutes")
            await self._log("INFO", f"Trials: {len(self._outcomes)} completed")
            await self._log("INFO", f"Best objective: {best_value:.2f}")
            await self._log("INFO", f"Best config: {best_params}")
            if best_outcome and best_outcome.metrics:
                await self._log(
                    "INFO", f"Best TPS: {best_outcome.metrics.get('throughput_tps', 0):.1f}"
                )
            await self._log("INFO", "=" * 50)

            # Update progress to show completion
            await self._update_progress(
                step=self.n_trials,
                total=self.n_trials,
                step_name="Completed",
                params=best_params,
            )

            await self._update_status(TuningJobStatus.COMPLETED, "Tuning completed successfully")
            self.job.completed_at = datetime.now(UTC)
            await self.db.commit()

            return {
                "success": True,
                "best_config": best_params,
                "best_value": best_value,
                "trials_completed": len(self._outcomes),
            }

        except Exception as e:
            await self._log("ERROR", f"Tuning failed: {e}")
            logger.exception(f"[Job #{self.job.id}] Tuning failed: {e}")
            await self._update_status(TuningJobStatus.FAILED, f"Error: {str(e)}")
            return {"success": False, "error": str(e)}

        finally:
            # Cleanup any remaining deployment
            await self._cleanup_deployment()

    async def _analyze_hardware(self) -> HardwareProfile:
        """Analyze worker hardware configuration"""
        result = await self.db.execute(select(Worker).where(Worker.id == self.job.worker_id))
        worker = result.scalar_one_or_none()

        if not worker:
            raise RuntimeError(f"Worker {self.job.worker_id} not found")

        return HardwareProfile.from_worker(worker)

    async def _query_knowledge_base(self, hardware: HardwareProfile) -> list[dict[str, Any]]:
        """Query historical results for similar configurations"""
        model = self.job.model
        model_family = self._extract_model_family(model.name)

        # Query for similar hardware + model combinations
        stmt = (
            select(PerformanceKnowledge)
            .where(PerformanceKnowledge.gpu_model.ilike(f"%{hardware.gpu_name.split()[0]}%"))
            .where(PerformanceKnowledge.model_family == model_family)
            .order_by(PerformanceKnowledge.score.desc().nulls_last())
            .limit(5)
        )

        result = await self.db.execute(stmt)
        records = result.scalars().all()

        warm_start_params = []
        for r in records:
            params = {
                "engine": r.engine,
                "gpu_memory_utilization": 0.9,  # Default
                "max_num_seqs": 16,  # Default
            }
            if r.tensor_parallel and r.tensor_parallel > 1:
                params["tensor_parallel_size"] = r.tensor_parallel
            warm_start_params.append(params)

        return warm_start_params

    def _estimate_model_size(self) -> float:
        """Estimate model size in GB from name"""
        name_lower = self.job.model.name.lower()

        # Extract number of parameters from common patterns
        import re

        patterns = [
            r"(\d+\.?\d*)b",  # 7b, 7.5b
            r"(\d+)b-",  # 7b-
            r"-(\d+)b",  # -7b
        ]

        for pattern in patterns:
            match = re.search(pattern, name_lower)
            if match:
                params_b = float(match.group(1))
                # Rough estimate: 2 bytes per param (FP16)
                return params_b * 2

        # Default for unknown models
        return 7.0

    def _extract_model_family(self, name: str) -> str:
        """Extract model family from name"""
        name_lower = name.lower()
        families = {
            "qwen": "Qwen",
            "llama": "Llama",
            "mistral": "Mistral",
            "deepseek": "DeepSeek",
            "phi": "Phi",
            "gemma": "Gemma",
        }
        for key, value in families.items():
            if key in name_lower:
                return value
        return "Unknown"

    async def _execute_trial(self, trial_idx: int, params: dict[str, Any]) -> TrialOutcome:
        """Execute a single trial: deploy, benchmark, cleanup"""
        start_time = time.time()
        outcome = TrialOutcome(trial_id=trial_idx, parameters=params.copy())

        try:
            # Step 1: Deploy model
            await self._log("INFO", "Deploying model...")
            await self._update_status(
                TuningJobStatus.BENCHMARKING, f"Trial {trial_idx + 1}: Deploying model..."
            )

            deployment_id = await self._deploy_model(params)
            outcome.deployment_id = deployment_id
            self._current_deployment_id = deployment_id
            await self._log("INFO", f"Deployment #{deployment_id} created")

            # Step 2: Wait for deployment
            await self._log("INFO", "Waiting for model to load...")
            await self._update_status(
                TuningJobStatus.BENCHMARKING, f"Trial {trial_idx + 1}: Waiting for model to load..."
            )

            ready = await self._wait_for_deployment(deployment_id, timeout=self.timeout_per_trial)
            if not ready:
                outcome.success = False
                outcome.error_message = "Deployment timeout"
                await self._log("ERROR", f"Deployment timeout after {self.timeout_per_trial}s")
                return outcome

            await self._log("INFO", "Model loaded successfully")

            # Step 3: Run benchmark
            await self._log("INFO", "Running benchmark...")
            await self._update_status(
                TuningJobStatus.BENCHMARKING, f"Trial {trial_idx + 1}: Running benchmark..."
            )

            metrics = await self._run_benchmark(deployment_id)
            outcome.metrics = metrics
            await self._log(
                "INFO", f"Benchmark complete: TPS={metrics.get('throughput_tps', 0):.1f}"
            )

        except Exception as e:
            outcome.success = False
            outcome.error_message = str(e)
            await self._log("ERROR", f"Trial exception: {e}")

        finally:
            # Step 4: Cleanup
            await self._log("INFO", "Cleaning up deployment...")
            await self._cleanup_deployment()
            outcome.duration_seconds = time.time() - start_time
            await self._log("INFO", f"Trial duration: {outcome.duration_seconds:.1f}s")

        return outcome

    async def _deploy_model(self, params: dict[str, Any]) -> int:
        """Deploy model with given parameters"""
        engine = params.get("engine", "vllm")
        gpu_indexes = list(range(params.get("tensor_parallel_size", 1)))

        # Build extra params from tuning parameters
        # Note: vLLM and SGLang use different parameter names
        extra_params = {}

        if engine == "sglang":
            # SGLang parameter names
            if "gpu_memory_utilization" in params:
                extra_params["mem-fraction-static"] = params["gpu_memory_utilization"]
            if "max_num_seqs" in params:
                extra_params["max-running-requests"] = params["max_num_seqs"]
        else:
            # vLLM parameter names (default)
            if "gpu_memory_utilization" in params:
                extra_params["gpu-memory-utilization"] = params["gpu_memory_utilization"]
            if "max_num_seqs" in params:
                extra_params["max-num-seqs"] = params["max_num_seqs"]

        # Create deployment
        deployment = Deployment(
            name=f"tuning-trial-{self.job.id}-{int(time.time())}",
            model_id=self.job.model_id,
            worker_id=self.job.worker_id,
            backend=engine,
            gpu_indexes=gpu_indexes,
            extra_params=extra_params,
            status=DeploymentStatus.PENDING.value,
        )

        self.db.add(deployment)
        await self.db.commit()
        await self.db.refresh(deployment)

        # Start deployment async
        asyncio.create_task(self.deployer.deploy(deployment.id))

        return deployment.id

    async def _wait_for_deployment(self, deployment_id: int, timeout: int = 600) -> bool:
        """Wait for deployment to be ready"""
        start = time.time()

        while time.time() - start < timeout:
            if self._cancelled:
                return False

            # Expire all cached objects to get fresh data from database
            self.db.expire_all()

            result = await self.db.execute(select(Deployment).where(Deployment.id == deployment_id))
            deployment = result.scalar_one_or_none()

            if not deployment:
                await self._log("WARNING", f"Deployment #{deployment_id} not found")
                return False

            await self._log("INFO", f"Deployment status: {deployment.status}")

            if deployment.status == DeploymentStatus.RUNNING.value:
                return True

            if deployment.status in [DeploymentStatus.ERROR.value, DeploymentStatus.STOPPED.value]:
                await self._log("ERROR", f"Deployment failed with status: {deployment.status}")
                return False

            await asyncio.sleep(5)

        return False

    async def _run_benchmark(
        self,
        deployment_id: int,
        num_requests: int = 20,
        concurrency: int = 4,
    ) -> dict[str, float]:
        """Run benchmark against deployment"""
        result = await self.db.execute(
            select(Deployment)
            .where(Deployment.id == deployment_id)
            .options(selectinload(Deployment.worker), selectinload(Deployment.model))
        )
        deployment = result.scalar_one_or_none()

        if not deployment or deployment.status != DeploymentStatus.RUNNING.value:
            raise RuntimeError("Deployment not running")

        # Build endpoint URL
        worker_ip = deployment.worker.address.split(":")[0]
        base_url = f"http://{worker_ip}:{deployment.port}/v1"
        model_name = deployment.model.model_id

        # Run HTTP benchmark (reuse existing implementation pattern)
        return await self._http_benchmark(base_url, model_name, num_requests, concurrency)

    async def _http_benchmark(
        self,
        base_url: str,
        model_name: str,
        num_requests: int,
        concurrency: int,
    ) -> dict[str, float]:
        """Execute HTTP benchmark against OpenAI-compatible endpoint"""
        test_prompt = "Explain the concept of machine learning in simple terms. " * 20

        results = []
        semaphore = asyncio.Semaphore(concurrency)

        async def make_request(client: httpx.AsyncClient) -> dict | None:
            async with semaphore:
                start = time.perf_counter()
                first_token_time = None
                token_count = 0

                try:
                    async with client.stream(
                        "POST",
                        f"{base_url}/chat/completions",
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": test_prompt}],
                            "max_tokens": 64,
                            "stream": True,
                        },
                        timeout=60.0,
                    ) as resp:
                        if resp.status_code != 200:
                            return None

                        async for line in resp.aiter_lines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    chunk = json.loads(line[6:])
                                    content = (
                                        chunk.get("choices", [{}])[0]
                                        .get("delta", {})
                                        .get("content", "")
                                    )
                                    if content:
                                        if first_token_time is None:
                                            first_token_time = time.perf_counter()
                                        token_count += 1
                                except json.JSONDecodeError:
                                    pass

                        end = time.perf_counter()

                        if first_token_time and token_count > 0:
                            return {
                                "ttft_ms": (first_token_time - start) * 1000,
                                "tpot_ms": (
                                    ((end - first_token_time) / max(1, token_count - 1)) * 1000
                                    if token_count > 1
                                    else 0
                                ),
                                "tokens": token_count,
                                "total_time": end - start,
                            }
                except Exception:
                    pass
                return None

        async with httpx.AsyncClient() as client:
            # Warmup
            for _ in range(2):
                await make_request(client)

            # Actual benchmark
            tasks = [make_request(client) for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)

        valid = [r for r in results if r]
        if not valid:
            return {"throughput_tps": 0, "avg_ttft_ms": 0, "avg_tpot_ms": 0}

        total_tokens = sum(r["tokens"] for r in valid)
        total_time = sum(r["total_time"] for r in valid)

        return {
            "throughput_tps": round(total_tokens / total_time, 2) if total_time > 0 else 0,
            "avg_ttft_ms": round(sum(r["ttft_ms"] for r in valid) / len(valid), 2),
            "avg_tpot_ms": round(
                sum(r["tpot_ms"] for r in valid if r["tpot_ms"] > 0)
                / max(1, len([r for r in valid if r["tpot_ms"] > 0])),
                2,
            ),
            "successful_requests": len(valid),
            "total_requests": num_requests,
        }

    async def _cleanup_deployment(self):
        """Stop and remove current deployment"""
        if self._current_deployment_id:
            try:
                result = await self.db.execute(
                    select(Deployment).where(Deployment.id == self._current_deployment_id)
                )
                deployment = result.scalar_one_or_none()

                if deployment:
                    await self.deployer.stop(deployment.id)
                    deployment.status = DeploymentStatus.STOPPED.value
                    await self.db.delete(deployment)
                    await self.db.commit()
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
            finally:
                self._current_deployment_id = None

    async def _save_to_knowledge_base(self, params: dict[str, Any], outcome: TrialOutcome | None):
        """Save best result to knowledge base for future warm-start"""
        if not outcome or not outcome.metrics:
            return

        model = self.job.model
        worker_result = await self.db.execute(select(Worker).where(Worker.id == self.job.worker_id))
        worker = worker_result.scalar_one_or_none()

        if not worker:
            return

        hardware = HardwareProfile.from_worker(worker)

        knowledge = PerformanceKnowledge(
            gpu_model=hardware.gpu_name,
            gpu_count=hardware.gpu_count,
            total_vram_gb=hardware.total_vram_gb,
            model_name=model.name,
            model_family=self._extract_model_family(model.name),
            engine=params.get("engine", "vllm"),
            tensor_parallel=params.get("tensor_parallel_size", 1),
            extra_args=params,
            throughput_tps=outcome.metrics.get("throughput_tps", 0),
            ttft_ms=outcome.metrics.get("avg_ttft_ms", 0),
            tpot_ms=outcome.metrics.get("avg_tpot_ms", 0),
            score=outcome.metrics.get("throughput_tps", 0),
            source_tuning_job_id=self.job.id,
        )

        self.db.add(knowledge)
        await self.db.commit()

    async def _update_status(self, status: TuningJobStatus, message: str):
        """Update job status and message"""
        self.job.status = status.value
        self.job.status_message = message
        await self.db.commit()

    async def _update_progress(
        self,
        step: int,
        total: int,
        step_name: str,
        params: dict[str, Any] | None = None,
    ):
        """Update job progress"""
        self.job.current_step = step
        self.job.total_steps = total
        self.job.progress = {
            "step": step,
            "total_steps": total,
            "step_name": step_name,
            "current_config": params,
            "completed_trials": len(self._outcomes),
            "successful_trials": len([o for o in self._outcomes if o.success]),
        }
        await self.db.commit()


async def run_bayesian_tuning(job_id: int, n_trials: int = 10):
    """
    Entry point for running Bayesian optimization tuning.

    Args:
        job_id: TuningJob ID
        n_trials: Number of optimization trials
    """
    async with async_session_maker() as db:
        result = await db.execute(
            select(TuningJob)
            .where(TuningJob.id == job_id)
            .options(
                selectinload(TuningJob.model),
                selectinload(TuningJob.worker),
            )
        )
        job = result.scalar_one_or_none()

        if not job:
            logger.error(f"Tuning job {job_id} not found")
            return

        service = BayesianTuningService(db, job, n_trials=n_trials)
        await service.run()

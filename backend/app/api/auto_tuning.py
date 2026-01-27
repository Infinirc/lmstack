"""Auto-Tuning API routes

Implements the Auto-Tuning Agent workflow:
1. Environment Analysis - Query hardware and model info
2. Knowledge Base Query - Search for similar configurations
3. Configuration Space Exploration - Generate candidate configs
4. Auto Benchmark - Test each configuration
5. Result Analysis - Recommend best configuration
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.deps import require_operator, require_viewer
from app.database import get_db
from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import LLMModel
from app.models.tuning import (
    BenchmarkResult,
    OptimizationTarget,
    PerformanceKnowledge,
    TuningJob,
    TuningJobStatus,
)
from app.models.user import User
from app.models.worker import Worker
from app.schemas.tuning import (
    BenchmarkMetrics,
    BenchmarkRequest,
    BenchmarkResultListResponse,
    BenchmarkResultResponse,
    KnowledgeQuery,
    KnowledgeQueryResponse,
    KnowledgeRecord,
    KnowledgeSaveRequest,
    TuningJobCreate,
    TuningJobListResponse,
    TuningJobProgress,
    TuningJobResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Helper Functions
# ============================================================================


def tuning_job_to_response(job: TuningJob, include_conversation: bool = True) -> TuningJobResponse:
    """Convert tuning job model to response schema"""
    progress = None
    if job.progress:
        progress = TuningJobProgress(**job.progress)

    # Parse conversation log
    conversation_log = None
    if include_conversation and job.conversation_log:
        from app.schemas.tuning import ConversationMessage

        conversation_log = [ConversationMessage(**msg) for msg in job.conversation_log]

    # Parse logs
    logs = None
    if job.logs:
        from app.schemas.tuning import TuningLogEntry

        logs = [TuningLogEntry(**log) for log in job.logs]

    return TuningJobResponse(
        id=job.id,
        model_id=job.model_id,
        worker_id=job.worker_id,
        optimization_target=job.optimization_target,
        status=job.status,
        status_message=job.status_message,
        current_step=job.current_step,
        total_steps=job.total_steps,
        progress=progress,
        best_config=job.best_config,
        all_results=job.all_results,
        logs=logs,
        conversation_log=conversation_log,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        model_name=job.model.name if job.model else None,
        worker_name=job.worker.name if job.worker else None,
    )


def benchmark_result_to_response(result: BenchmarkResult) -> BenchmarkResultResponse:
    """Convert benchmark result model to response schema"""
    return BenchmarkResultResponse(
        id=result.id,
        tuning_job_id=result.tuning_job_id,
        deployment_id=result.deployment_id,
        config=result.config,
        test_type=result.test_type,
        test_duration_seconds=result.test_duration_seconds,
        input_length=result.input_length,
        output_length=result.output_length,
        concurrency=result.concurrency,
        metrics=BenchmarkMetrics(
            throughput_tps=result.throughput_tps,
            ttft_ms=result.ttft_ms,
            tpot_ms=result.tpot_ms,
            total_latency_ms=result.total_latency_ms,
            gpu_utilization=result.gpu_utilization,
            vram_usage_gb=result.vram_usage_gb,
        ),
        error_message=result.error_message,
        created_at=result.created_at,
    )


# ============================================================================
# Tuning Job Endpoints
# ============================================================================


@router.get("/jobs", response_model=TuningJobListResponse)
async def list_tuning_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: TuningJobStatus | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """List all tuning jobs"""
    query = select(TuningJob).options(
        selectinload(TuningJob.model),
        selectinload(TuningJob.worker),
    )

    if status:
        query = query.where(TuningJob.status == status.value)

    # Count
    count_query = select(func.count()).select_from(
        select(TuningJob).where(*([TuningJob.status == status.value] if status else [])).subquery()
    )
    total = await db.scalar(count_query)

    # Get results
    query = query.offset(skip).limit(limit).order_by(TuningJob.created_at.desc())
    result = await db.execute(query)
    jobs = result.scalars().all()

    return TuningJobListResponse(
        items=[tuning_job_to_response(j, include_conversation=False) for j in jobs],
        total=total or 0,
    )


@router.post("/jobs", response_model=TuningJobResponse, status_code=201)
async def create_tuning_job(
    job_in: TuningJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Create a new auto-tuning job"""
    # Verify model exists
    model_result = await db.execute(select(LLMModel).where(LLMModel.id == job_in.model_id))
    model = model_result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Verify worker exists
    worker_result = await db.execute(select(Worker).where(Worker.id == job_in.worker_id))
    worker = worker_result.scalar_one_or_none()
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Create tuning job
    job = TuningJob(
        model_id=job_in.model_id,
        worker_id=job_in.worker_id,
        optimization_target=job_in.optimization_target.value,
        status=TuningJobStatus.PENDING.value,
        progress={
            "step": 0,
            "total_steps": 5,
            "step_name": "Initializing",
            "step_description": "Preparing auto-tuning job...",
            "configs_tested": 0,
            "configs_total": 0,
        },
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Prepare LLM config for agent
    llm_config = None
    if job_in.llm_config:
        llm_config = job_in.llm_config.model_dump()

    # Start tuning in background
    background_tasks.add_task(run_auto_tuning, job.id, llm_config)

    # Reload with relationships
    result = await db.execute(
        select(TuningJob)
        .where(TuningJob.id == job.id)
        .options(
            selectinload(TuningJob.model),
            selectinload(TuningJob.worker),
        )
    )
    job = result.scalar_one()

    return tuning_job_to_response(job)


@router.get("/jobs/{job_id}", response_model=TuningJobResponse)
async def get_tuning_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Get a tuning job by ID"""
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
        raise HTTPException(status_code=404, detail="Tuning job not found")

    return tuning_job_to_response(job)


@router.post("/jobs/{job_id}/cancel", response_model=TuningJobResponse)
async def cancel_tuning_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Cancel a running tuning job"""
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
        raise HTTPException(status_code=404, detail="Tuning job not found")

    if job.status in [TuningJobStatus.COMPLETED.value, TuningJobStatus.FAILED.value]:
        raise HTTPException(status_code=400, detail="Job is already finished")

    job.status = TuningJobStatus.CANCELLED.value
    job.status_message = "Cancelled by user"
    await db.commit()
    await db.refresh(job)

    return tuning_job_to_response(job)


@router.delete("/jobs/{job_id}")
async def delete_tuning_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Delete a tuning job"""
    result = await db.execute(select(TuningJob).where(TuningJob.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Tuning job not found")

    # Don't allow deleting running jobs
    if job.status in ["pending", "analyzing", "querying_kb", "exploring", "benchmarking"]:
        raise HTTPException(status_code=400, detail="Cannot delete a running job. Cancel it first.")

    await db.delete(job)
    await db.commit()

    return {"success": True, "message": f"Tuning job {job_id} deleted"}


# ============================================================================
# Benchmark Endpoints
# ============================================================================


@router.post("/benchmarks/run", response_model=BenchmarkResultResponse)
async def run_benchmark(
    request: BenchmarkRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Run a standalone benchmark on a deployment"""
    # Verify deployment exists and is running
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == request.deployment_id)
        .options(selectinload(Deployment.model))
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if deployment.status != DeploymentStatus.RUNNING.value:
        raise HTTPException(status_code=400, detail="Deployment is not running")

    # Run benchmark
    metrics = await _run_benchmark_test(deployment, request)

    # Save result
    benchmark_result = BenchmarkResult(
        deployment_id=deployment.id,
        config={
            "engine": deployment.backend,
            "gpu_indexes": deployment.gpu_indexes,
            "extra_params": deployment.extra_params,
        },
        test_type=request.test_type,
        test_duration_seconds=request.duration_seconds,
        input_length=request.input_length,
        output_length=request.output_length,
        concurrency=request.concurrency,
        throughput_tps=metrics.get("throughput_tps"),
        ttft_ms=metrics.get("ttft_ms"),
        tpot_ms=metrics.get("tpot_ms"),
        total_latency_ms=metrics.get("total_latency_ms"),
        gpu_utilization=metrics.get("gpu_utilization"),
        vram_usage_gb=metrics.get("vram_usage_gb"),
        raw_results=metrics.get("raw"),
        error_message=metrics.get("error"),
    )

    db.add(benchmark_result)
    await db.commit()
    await db.refresh(benchmark_result)

    return benchmark_result_to_response(benchmark_result)


@router.get("/benchmarks", response_model=BenchmarkResultListResponse)
async def list_benchmark_results(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    deployment_id: int | None = None,
    tuning_job_id: int | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """List benchmark results"""
    query = select(BenchmarkResult)

    if deployment_id:
        query = query.where(BenchmarkResult.deployment_id == deployment_id)
    if tuning_job_id:
        query = query.where(BenchmarkResult.tuning_job_id == tuning_job_id)

    # Count
    count_query = select(func.count()).select_from(
        select(BenchmarkResult)
        .where(
            *([BenchmarkResult.deployment_id == deployment_id] if deployment_id else []),
            *([BenchmarkResult.tuning_job_id == tuning_job_id] if tuning_job_id else []),
        )
        .subquery()
    )
    total = await db.scalar(count_query)

    # Get results
    query = query.offset(skip).limit(limit).order_by(BenchmarkResult.created_at.desc())
    result = await db.execute(query)
    results = result.scalars().all()

    return BenchmarkResultListResponse(
        items=[benchmark_result_to_response(r) for r in results],
        total=total or 0,
    )


# ============================================================================
# Knowledge Base Endpoints
# ============================================================================


@router.post("/knowledge/query", response_model=KnowledgeQueryResponse)
async def query_knowledge_base(
    query: KnowledgeQuery,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Query the performance knowledge base for similar configurations"""
    stmt = select(PerformanceKnowledge)

    if query.model_name:
        stmt = stmt.where(PerformanceKnowledge.model_name.ilike(f"%{query.model_name}%"))
    if query.model_family:
        stmt = stmt.where(PerformanceKnowledge.model_family == query.model_family)
    if query.gpu_model:
        stmt = stmt.where(PerformanceKnowledge.gpu_model.ilike(f"%{query.gpu_model}%"))
    if query.min_vram_gb:
        stmt = stmt.where(PerformanceKnowledge.total_vram_gb >= query.min_vram_gb)

    # Order by score (computed based on optimization target)
    if query.optimization_target == OptimizationTarget.THROUGHPUT:
        stmt = stmt.order_by(PerformanceKnowledge.throughput_tps.desc())
    elif query.optimization_target == OptimizationTarget.LATENCY:
        stmt = stmt.order_by(PerformanceKnowledge.ttft_ms.asc())
    else:
        # Balanced - order by a combined score
        stmt = stmt.order_by(PerformanceKnowledge.score.desc().nulls_last())

    stmt = stmt.limit(query.limit)

    result = await db.execute(stmt)
    records = result.scalars().all()

    # Count total matches
    count_stmt = select(func.count()).select_from(PerformanceKnowledge)
    if query.model_name:
        count_stmt = count_stmt.where(
            PerformanceKnowledge.model_name.ilike(f"%{query.model_name}%")
        )
    if query.model_family:
        count_stmt = count_stmt.where(PerformanceKnowledge.model_family == query.model_family)
    if query.gpu_model:
        count_stmt = count_stmt.where(PerformanceKnowledge.gpu_model.ilike(f"%{query.gpu_model}%"))
    if query.min_vram_gb:
        count_stmt = count_stmt.where(PerformanceKnowledge.total_vram_gb >= query.min_vram_gb)

    total = await db.scalar(count_stmt)

    return KnowledgeQueryResponse(
        items=[KnowledgeRecord.model_validate(r) for r in records],
        total=total or 0,
        query=query,
    )


@router.post("/knowledge/save", response_model=KnowledgeRecord)
async def save_to_knowledge_base(
    request: KnowledgeSaveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_operator),
):
    """Save a benchmark result to the knowledge base"""
    # Get benchmark result with related data
    result = await db.execute(
        select(BenchmarkResult)
        .where(BenchmarkResult.id == request.benchmark_result_id)
        .options(selectinload(BenchmarkResult.deployment))
    )
    benchmark = result.scalar_one_or_none()

    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark result not found")

    if not benchmark.throughput_tps or not benchmark.ttft_ms or not benchmark.tpot_ms:
        raise HTTPException(status_code=400, detail="Benchmark result has incomplete metrics")

    # Get deployment and worker info
    deployment = benchmark.deployment
    worker_result = await db.execute(select(Worker).where(Worker.id == deployment.worker_id))
    worker = worker_result.scalar_one_or_none()

    model_result = await db.execute(select(LLMModel).where(LLMModel.id == deployment.model_id))
    model = model_result.scalar_one_or_none()

    if not worker or not model:
        raise HTTPException(status_code=400, detail="Missing worker or model info")

    # Extract GPU info from worker
    gpu_info = worker.gpu_info or []
    gpu_model = gpu_info[0].get("name", "Unknown") if gpu_info else "Unknown"
    gpu_count = len(deployment.gpu_indexes) if deployment.gpu_indexes else len(gpu_info)
    total_vram = sum(g.get("memory_total", 0) for g in gpu_info) / 1024  # Convert to GB

    # Compute score (balanced)
    # Higher throughput is better, lower latency is better
    # Normalize and combine
    score = benchmark.throughput_tps / (benchmark.ttft_ms + benchmark.tpot_ms * 100)

    # Create knowledge record
    record = PerformanceKnowledge(
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        total_vram_gb=total_vram,
        model_name=model.name,
        model_family=request.model_family,
        model_params_b=request.model_params_b,
        engine=deployment.backend,
        quantization=benchmark.config.get("quantization"),
        tensor_parallel=len(deployment.gpu_indexes) if deployment.gpu_indexes else 1,
        extra_args=deployment.extra_params,
        throughput_tps=benchmark.throughput_tps,
        ttft_ms=benchmark.ttft_ms,
        tpot_ms=benchmark.tpot_ms,
        gpu_utilization=benchmark.gpu_utilization,
        vram_usage_gb=benchmark.vram_usage_gb,
        test_dataset="synthetic",
        input_length=benchmark.input_length,
        output_length=benchmark.output_length,
        concurrency=benchmark.concurrency,
        score=score,
        source_tuning_job_id=benchmark.tuning_job_id,
    )

    db.add(record)
    await db.commit()
    await db.refresh(record)

    return KnowledgeRecord.model_validate(record)


# ============================================================================
# Agent Chat Endpoint
# ============================================================================


class AgentChatRequest(BaseModel):
    """Request for agent chat"""

    message: str
    config: dict
    history: list[dict] = []


class AgentChatResponse(BaseModel):
    """Response from agent chat"""

    content: str
    tool_calls: list[dict] | None = None


@router.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(
    request: AgentChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Chat with the Auto-Tuning Agent"""
    from app.services.tuning_agent import AGENT_SYSTEM_PROMPT, AgentToolExecutor, get_agent_tools

    config = request.config
    provider = config.get("provider", "system")

    # Build client based on provider
    if provider == "system":
        # Use a system deployment
        deployment_id = config.get("deploymentId")
        if not deployment_id:
            raise HTTPException(status_code=400, detail="No deployment selected")

        result = await db.execute(
            select(Deployment)
            .where(Deployment.id == deployment_id)
            .options(selectinload(Deployment.worker))
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        if deployment.status != DeploymentStatus.RUNNING.value:
            raise HTTPException(status_code=400, detail="Deployment is not running")

        worker = deployment.worker
        base_url = f"http://{worker.host}:{deployment.port}/v1"
        api_key = "dummy"
        model = "default"

    elif provider == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = config.get("apiKey")
        model = config.get("model", "gpt-4o")

    elif provider == "anthropic":
        # Anthropic uses different API format, need adapter
        raise HTTPException(
            status_code=400, detail="Anthropic not yet supported, use OpenAI-compatible endpoint"
        )

    elif provider == "custom":
        base_url = config.get("baseUrl")
        api_key = config.get("apiKey", "dummy")
        model = config.get("model", "default")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    # Build messages
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    # Add history
    for msg in request.history[-10:]:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": request.message})

    # Create dummy job for tool executor
    class DummyJob:
        id = 0
        model_id = 0
        worker_id = 0

    executor = AgentToolExecutor(db, DummyJob())

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Call LLM with tools
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=get_agent_tools(),
            tool_choice="auto",
            max_tokens=4096,
        )

        assistant_message = response.choices[0].message
        content = assistant_message.content or ""
        tool_calls_result = []

        # Execute tool calls if any
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                import json

                tool_args = json.loads(tool_call.function.arguments)

                # Execute tool
                result = await executor.execute(tool_name, tool_args)

                tool_calls_result.append(
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "result": (
                            json.loads(result)
                            if result.startswith("{") or result.startswith("[")
                            else result
                        ),
                    }
                )

            # If there were tool calls but no content, generate a summary
            if not content and tool_calls_result:
                content = f"I executed {len(tool_calls_result)} tool(s). See the results below."

        return AgentChatResponse(
            content=content,
            tool_calls=tool_calls_result if tool_calls_result else None,
        )

    except Exception as e:
        logger.exception(f"Agent chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# ============================================================================
# Auto-Tuning Runner
# ============================================================================


async def run_auto_tuning(job_id: int, llm_config: dict | None = None):
    """Run the Auto-Tuning process using Bayesian optimization.

    Uses Optuna's TPE (Tree-structured Parzen Estimator) for efficient
    hyperparameter search instead of LLM Agent.

    Args:
        job_id: The tuning job ID
        llm_config: Legacy parameter (ignored, kept for API compatibility)
    """
    from app.services.bayesian_tuner import run_bayesian_tuning

    # Default to 10 trials for good optimization coverage
    await run_bayesian_tuning(job_id, n_trials=10)


async def _run_benchmark_test(deployment: Deployment, request: BenchmarkRequest) -> dict:
    """Run actual benchmark test on a deployment using HTTP requests"""
    from app.services.tuning_agent import _run_http_benchmark

    # Get worker info
    worker = deployment.worker
    if not worker:
        return {"error": "Worker not found"}

    base_url = f"http://{worker.host}:{deployment.port}/v1"

    result = await _run_http_benchmark(
        base_url=base_url,
        num_requests=max(10, request.concurrency * 5),
        concurrency=request.concurrency,
        input_tokens=request.input_length,
        output_tokens=request.output_length,
    )

    if not result.get("success"):
        return {"error": result.get("error", "Benchmark failed")}

    metrics = result.get("metrics", {})
    return {
        "throughput_tps": metrics.get("throughput_tps"),
        "ttft_ms": metrics.get("avg_ttft_ms"),
        "tpot_ms": metrics.get("avg_tpot_ms"),
        "total_latency_ms": None,  # Not directly measured
        "gpu_utilization": None,  # Would need GPU monitoring
        "vram_usage_gb": None,  # Would need GPU monitoring
        "raw": result.get("summary"),
    }

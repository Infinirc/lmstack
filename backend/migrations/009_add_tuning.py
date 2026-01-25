"""
Migration: Add auto-tuning tables

This migration creates tables for auto-tuning, benchmarks, and performance knowledge base.

Run with: python -m migrations.009_add_tuning
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings


async def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists (SQLite compatible)"""
    result = await conn.execute(
        text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    )
    return result.fetchone() is not None


async def migrate():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=True)

    async with engine.begin() as conn:
        # Create tuning_jobs table
        if not await table_exists(conn, "tuning_jobs"):
            print("Creating 'tuning_jobs' table...")
            await conn.execute(
                text(
                    """
                CREATE TABLE tuning_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    worker_id INTEGER NOT NULL,
                    optimization_target VARCHAR(50) DEFAULT 'balanced',
                    status VARCHAR(50) DEFAULT 'pending',
                    status_message TEXT,
                    current_step INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 5,
                    progress JSON,
                    best_config JSON,
                    all_results JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME,
                    FOREIGN KEY (model_id) REFERENCES llm_models(id),
                    FOREIGN KEY (worker_id) REFERENCES workers(id)
                )
            """
                )
            )
            print("'tuning_jobs' table created successfully!")
        else:
            print("'tuning_jobs' table already exists")

        # Create benchmark_results table
        if not await table_exists(conn, "benchmark_results"):
            print("Creating 'benchmark_results' table...")
            await conn.execute(
                text(
                    """
                CREATE TABLE benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tuning_job_id INTEGER,
                    deployment_id INTEGER NOT NULL,
                    config JSON NOT NULL,
                    test_type VARCHAR(50) DEFAULT 'throughput',
                    test_duration_seconds INTEGER DEFAULT 60,
                    input_length INTEGER DEFAULT 512,
                    output_length INTEGER DEFAULT 128,
                    concurrency INTEGER DEFAULT 1,
                    throughput_tps REAL,
                    ttft_ms REAL,
                    tpot_ms REAL,
                    total_latency_ms REAL,
                    gpu_utilization REAL,
                    vram_usage_gb REAL,
                    raw_results JSON,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tuning_job_id) REFERENCES tuning_jobs(id),
                    FOREIGN KEY (deployment_id) REFERENCES deployments(id)
                )
            """
                )
            )
            print("'benchmark_results' table created successfully!")
        else:
            print("'benchmark_results' table already exists")

        # Create performance_knowledge table
        if not await table_exists(conn, "performance_knowledge"):
            print("Creating 'performance_knowledge' table...")
            await conn.execute(
                text(
                    """
                CREATE TABLE performance_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gpu_model VARCHAR(255) NOT NULL,
                    gpu_count INTEGER DEFAULT 1,
                    total_vram_gb REAL NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    model_family VARCHAR(100) NOT NULL,
                    model_params_b REAL,
                    engine VARCHAR(50) NOT NULL,
                    quantization VARCHAR(50),
                    tensor_parallel INTEGER DEFAULT 1,
                    extra_args JSON,
                    throughput_tps REAL NOT NULL,
                    ttft_ms REAL NOT NULL,
                    tpot_ms REAL NOT NULL,
                    gpu_utilization REAL,
                    vram_usage_gb REAL,
                    test_dataset VARCHAR(100) DEFAULT 'synthetic',
                    input_length INTEGER DEFAULT 512,
                    output_length INTEGER DEFAULT 128,
                    concurrency INTEGER DEFAULT 1,
                    score REAL,
                    source_tuning_job_id INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_tuning_job_id) REFERENCES tuning_jobs(id)
                )
            """
                )
            )
            print("'performance_knowledge' table created successfully!")
        else:
            print("'performance_knowledge' table already exists")

        # Create indexes for performance knowledge queries
        print("Creating indexes...")
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_model_family ON performance_knowledge(model_family)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_gpu_model ON performance_knowledge(gpu_model)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_engine ON performance_knowledge(engine)"
                )
            )
            await conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_tuning_jobs_status ON tuning_jobs(status)")
            )
            print("Indexes created successfully!")
        except Exception as e:
            print(f"Note: Some indexes may already exist: {e}")

        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("=" * 50)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

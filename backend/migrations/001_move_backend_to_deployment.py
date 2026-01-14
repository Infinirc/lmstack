"""
Migration: Move backend from LLMModel to Deployment

This migration:
1. Adds 'source' column to llm_models table (huggingface/ollama)
2. Migrates backend values to source (vllm/sglang -> huggingface, ollama -> ollama)
3. Adds 'backend' column to deployments table
4. Copies backend values from related models to deployments

Run with: python -m migrations.001_move_backend_to_deployment
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings


async def get_table_columns(conn, table_name: str) -> list[str]:
    """Get column names for a table (SQLite compatible)"""
    result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
    rows = result.fetchall()
    return [row[1] for row in rows]  # column name is at index 1


async def migrate():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=True)

    async with engine.begin() as conn:
        # Get current columns
        llm_model_columns = await get_table_columns(conn, 'llm_models')
        deployment_columns = await get_table_columns(conn, 'deployments')

        print(f"Current llm_models columns: {llm_model_columns}")
        print(f"Current deployments columns: {deployment_columns}")

        # Step 1: Add 'source' column to llm_models if not exists
        if 'source' not in llm_model_columns:
            print("Adding 'source' column to llm_models...")
            await conn.execute(text(
                "ALTER TABLE llm_models ADD COLUMN source VARCHAR(50) DEFAULT 'huggingface'"
            ))

            # Migrate backend values to source
            if 'backend' in llm_model_columns:
                print("Migrating backend values to source...")
                await conn.execute(text(
                    "UPDATE llm_models SET source = 'ollama' WHERE backend = 'ollama'"
                ))
                await conn.execute(text(
                    "UPDATE llm_models SET source = 'huggingface' WHERE backend IN ('vllm', 'sglang') OR backend IS NULL"
                ))
        else:
            print("'source' column already exists in llm_models")

        # Step 2: Add 'backend' column to deployments if not exists
        if 'backend' not in deployment_columns:
            print("Adding 'backend' column to deployments...")
            await conn.execute(text(
                "ALTER TABLE deployments ADD COLUMN backend VARCHAR(50) DEFAULT 'vllm'"
            ))

            # Copy backend values from models to deployments
            if 'backend' in llm_model_columns:
                print("Copying backend values from models to deployments...")
                await conn.execute(text("""
                    UPDATE deployments
                    SET backend = (
                        SELECT llm_models.backend
                        FROM llm_models
                        WHERE llm_models.id = deployments.model_id
                    )
                """))
        else:
            print("'backend' column already exists in deployments")

        # Note: SQLite doesn't support DROP COLUMN in older versions
        # The 'backend' column in llm_models will be ignored by the application
        # It can be removed manually later or by recreating the table

        print("\n" + "="*50)
        print("Migration completed successfully!")
        print("="*50)
        print("\nNote: The old 'backend' column in llm_models is kept for safety.")
        print("It will be ignored by the application.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

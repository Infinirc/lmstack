"""
Migration: Add Mac native deployment support to workers table

Adds os_type, gpu_type, and capabilities columns to track worker environment
and available backends (Docker, Ollama, MLX, llama.cpp).

Run with: python -m migrations.011_add_worker_mac_support
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings


async def column_exists(conn, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table (SQLite compatible)"""
    result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
    columns = [row[1] for row in result.fetchall()]
    return column_name in columns


async def migrate():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=True)

    async with engine.begin() as conn:
        # Add os_type column
        if not await column_exists(conn, "workers", "os_type"):
            print("Adding 'os_type' column to workers table...")
            await conn.execute(
                text(
                    """
                ALTER TABLE workers ADD COLUMN os_type VARCHAR(50) DEFAULT 'linux'
            """
                )
            )
            print("'os_type' column added!")
        else:
            print("'os_type' column already exists")

        # Add gpu_type column
        if not await column_exists(conn, "workers", "gpu_type"):
            print("Adding 'gpu_type' column to workers table...")
            await conn.execute(
                text(
                    """
                ALTER TABLE workers ADD COLUMN gpu_type VARCHAR(50) DEFAULT 'nvidia'
            """
                )
            )
            print("'gpu_type' column added!")
        else:
            print("'gpu_type' column already exists")

        # Add capabilities column (JSON)
        if not await column_exists(conn, "workers", "capabilities"):
            print("Adding 'capabilities' column to workers table...")
            await conn.execute(
                text(
                    """
                ALTER TABLE workers ADD COLUMN capabilities JSON
            """
                )
            )
            print("'capabilities' column added!")
        else:
            print("'capabilities' column already exists")

        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("=" * 50)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

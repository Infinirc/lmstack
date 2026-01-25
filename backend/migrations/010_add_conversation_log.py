"""
Migration: Add conversation_log column to tuning_jobs

This migration adds a JSON column to store the agent's conversation history.

Run with: python -m migrations.010_add_conversation_log
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
    """Check if a column exists (SQLite compatible)"""
    result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
    columns = result.fetchall()
    return any(col[1] == column_name for col in columns)


async def migrate():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=True)

    async with engine.begin() as conn:
        # Add conversation_log column to tuning_jobs
        if not await column_exists(conn, "tuning_jobs", "conversation_log"):
            print("Adding 'conversation_log' column to 'tuning_jobs' table...")
            await conn.execute(text("ALTER TABLE tuning_jobs ADD COLUMN conversation_log JSON"))
            print("'conversation_log' column added successfully!")
        else:
            print("'conversation_log' column already exists")

        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("=" * 50)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

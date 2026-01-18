"""
Migration: Add parent_app_id column to apps table for monitoring services

Run with: python -m migrations.008_add_app_parent_id
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
        # Add parent_app_id column if not exists
        if not await column_exists(conn, "apps", "parent_app_id"):
            print("Adding 'parent_app_id' column to apps table...")
            await conn.execute(
                text(
                    """
                ALTER TABLE apps ADD COLUMN parent_app_id INTEGER REFERENCES apps(id) ON DELETE CASCADE
            """
                )
            )
            print("'parent_app_id' column added successfully!")
        else:
            print("'parent_app_id' column already exists")

        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("=" * 50)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

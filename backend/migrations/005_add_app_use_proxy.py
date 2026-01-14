"""
Migration: Add use_proxy column to apps table

Run with: python -m migrations.005_add_app_use_proxy
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
    result = await conn.execute(text(
        f"PRAGMA table_info({table_name})"
    ))
    columns = [row[1] for row in result.fetchall()]
    return column_name in columns


async def migrate():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=True)

    async with engine.begin() as conn:
        # Add use_proxy column if not exists
        if not await column_exists(conn, 'apps', 'use_proxy'):
            print("Adding 'use_proxy' column to apps table...")
            await conn.execute(text("""
                ALTER TABLE apps ADD COLUMN use_proxy BOOLEAN DEFAULT 1
            """))
            print("'use_proxy' column added successfully!")
        else:
            print("'use_proxy' column already exists")

        print("\n" + "="*50)
        print("Migration completed successfully!")
        print("="*50)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

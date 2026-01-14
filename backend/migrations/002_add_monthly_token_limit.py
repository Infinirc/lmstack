"""
Migration: Add monthly_token_limit to api_keys

This migration adds the monthly_token_limit column to the api_keys table.

Run with: python -m migrations.002_add_monthly_token_limit
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
        api_key_columns = await get_table_columns(conn, "api_keys")

        print(f"Current api_keys columns: {api_key_columns}")

        # Add 'monthly_token_limit' column if not exists
        if "monthly_token_limit" not in api_key_columns:
            print("Adding 'monthly_token_limit' column to api_keys...")
            await conn.execute(text("ALTER TABLE api_keys ADD COLUMN monthly_token_limit INTEGER"))
            print("Column added successfully!")
        else:
            print("'monthly_token_limit' column already exists in api_keys")

        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("=" * 50)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

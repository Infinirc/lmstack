"""
Migration: Add apps table

This migration creates the apps table for deployed applications like Open WebUI.

Run with: python -m migrations.004_add_apps
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
    result = await conn.execute(text(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    ))
    return result.fetchone() is not None


async def migrate():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=True)

    async with engine.begin() as conn:
        # Create apps table if not exists
        if not await table_exists(conn, 'apps'):
            print("Creating 'apps' table...")
            await conn.execute(text("""
                CREATE TABLE apps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_type VARCHAR(50) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    worker_id INTEGER NOT NULL,
                    api_key_id INTEGER,
                    status VARCHAR(50) DEFAULT 'pending',
                    status_message TEXT,
                    container_id VARCHAR(255),
                    port INTEGER,
                    proxy_path VARCHAR(255) NOT NULL,
                    config JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (worker_id) REFERENCES workers(id),
                    FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
                )
            """))
            print("'apps' table created successfully!")
        else:
            print("'apps' table already exists")

        print("\n" + "="*50)
        print("Migration completed successfully!")
        print("="*50)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

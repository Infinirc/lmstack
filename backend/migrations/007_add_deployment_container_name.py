"""
Migration: Add container_name column to deployments table

This column stores the Docker container name for local deployments,
enabling internal Docker network communication (container-to-container).
Required for Windows Docker Desktop compatibility where host.docker.internal:port
doesn't work for backend-to-model communication.

Run with: python -m migrations.007_add_deployment_container_name
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
        # Add container_name column
        if not await column_exists(conn, "deployments", "container_name"):
            print("Adding 'container_name' column to deployments table...")
            await conn.execute(
                text(
                    """
                ALTER TABLE deployments ADD COLUMN container_name VARCHAR(255)
            """
                )
            )
            print("'container_name' column added!")
        else:
            print("'container_name' column already exists")

        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("=" * 50)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

"""
Migration: Add Tailscale/Headscale support to workers table

Run with: python -m migrations.006_add_worker_tailscale
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
        # Add connection_type column
        if not await column_exists(conn, 'workers', 'connection_type'):
            print("Adding 'connection_type' column to workers table...")
            await conn.execute(text("""
                ALTER TABLE workers ADD COLUMN connection_type VARCHAR(50) DEFAULT 'direct'
            """))
            print("'connection_type' column added!")
        else:
            print("'connection_type' column already exists")

        # Add tailscale_ip column
        if not await column_exists(conn, 'workers', 'tailscale_ip'):
            print("Adding 'tailscale_ip' column to workers table...")
            await conn.execute(text("""
                ALTER TABLE workers ADD COLUMN tailscale_ip VARCHAR(255)
            """))
            print("'tailscale_ip' column added!")
        else:
            print("'tailscale_ip' column already exists")

        # Add headscale_node_id column
        if not await column_exists(conn, 'workers', 'headscale_node_id'):
            print("Adding 'headscale_node_id' column to workers table...")
            await conn.execute(text("""
                ALTER TABLE workers ADD COLUMN headscale_node_id INTEGER
            """))
            print("'headscale_node_id' column added!")
        else:
            print("'headscale_node_id' column already exists")

        print("\n" + "="*50)
        print("Migration completed successfully!")
        print("="*50)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

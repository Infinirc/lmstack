"""
Migration: Add conversations and messages tables

This migration creates the conversations and messages tables for storing chat history.

Run with: python -m migrations.003_add_conversations
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
        text(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
    )
    return result.fetchone() is not None


async def migrate():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=True)

    async with engine.begin() as conn:
        # Create conversations table if not exists
        if not await table_exists(conn, "conversations"):
            print("Creating 'conversations' table...")
            await conn.execute(
                text(
                    """
                CREATE TABLE conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    deployment_id INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (deployment_id) REFERENCES deployments(id)
                )
            """
                )
            )
            # Create index on user_id
            await conn.execute(
                text("CREATE INDEX idx_conversations_user_id ON conversations(user_id)")
            )
            print("'conversations' table created successfully!")
        else:
            print("'conversations' table already exists")

        # Create messages table if not exists
        if not await table_exists(conn, "messages"):
            print("Creating 'messages' table...")
            await conn.execute(
                text(
                    """
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    thinking TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """
                )
            )
            # Create index on conversation_id
            await conn.execute(
                text(
                    "CREATE INDEX idx_messages_conversation_id ON messages(conversation_id)"
                )
            )
            print("'messages' table created successfully!")
        else:
            print("'messages' table already exists")

        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("=" * 50)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())

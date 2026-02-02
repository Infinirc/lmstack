"""Database configuration and session management"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all database models"""

    pass


async def get_db() -> AsyncSession:
    """Dependency for getting database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def _run_migrations(conn):
    """Run schema migrations for new columns (SQLite compatible)."""
    from sqlalchemy import text

    async def column_exists(table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
        columns = [row[1] for row in result.fetchall()]
        return column_name in columns

    # Migration: Add container_name to deployments (for Windows Docker compatibility)
    if not await column_exists("deployments", "container_name"):
        logger.info("Adding 'container_name' column to deployments table...")
        await conn.execute(text("ALTER TABLE deployments ADD COLUMN container_name VARCHAR(255)"))
        logger.info("'container_name' column added!")

    # Migration: Add is_local to registration_tokens (for local worker detection)
    if not await column_exists("registration_tokens", "is_local"):
        logger.info("Adding 'is_local' column to registration_tokens table...")
        await conn.execute(
            text("ALTER TABLE registration_tokens ADD COLUMN is_local BOOLEAN DEFAULT 0")
        )
        logger.info("'is_local' column added!")

    # Migration: Add conversation_type to conversations (for Agent chat support)
    if not await column_exists("conversations", "conversation_type"):
        logger.info("Adding 'conversation_type' column to conversations table...")
        await conn.execute(
            text(
                "ALTER TABLE conversations ADD COLUMN conversation_type VARCHAR(20) DEFAULT 'chat' NOT NULL"
            )
        )
        logger.info("'conversation_type' column added!")

    # Migration: Add agent_config to conversations (for Agent configuration)
    if not await column_exists("conversations", "agent_config"):
        logger.info("Adding 'agent_config' column to conversations table...")
        await conn.execute(text("ALTER TABLE conversations ADD COLUMN agent_config JSON"))
        logger.info("'agent_config' column added!")

    # Migration: Add tool_calls to messages (for Agent tool calls)
    if not await column_exists("messages", "tool_calls"):
        logger.info("Adding 'tool_calls' column to messages table...")
        await conn.execute(text("ALTER TABLE messages ADD COLUMN tool_calls JSON"))
        logger.info("'tool_calls' column added!")

    # Migration: Add tool_call_id to messages (for Agent tool results)
    if not await column_exists("messages", "tool_call_id"):
        logger.info("Adding 'tool_call_id' column to messages table...")
        await conn.execute(text("ALTER TABLE messages ADD COLUMN tool_call_id VARCHAR(100)"))
        logger.info("'tool_call_id' column added!")

    # Migration: Add step_type to messages (for Agent execution steps)
    if not await column_exists("messages", "step_type"):
        logger.info("Adding 'step_type' column to messages table...")
        await conn.execute(text("ALTER TABLE messages ADD COLUMN step_type VARCHAR(50)"))
        logger.info("'step_type' column added!")

    # Migration: Add execution_time_ms to messages (for tool execution timing)
    if not await column_exists("messages", "execution_time_ms"):
        logger.info("Adding 'execution_time_ms' column to messages table...")
        await conn.execute(text("ALTER TABLE messages ADD COLUMN execution_time_ms FLOAT"))
        logger.info("'execution_time_ms' column added!")

    # Migration: Add tuning_config to tuning_jobs (for multi-framework testing)
    if not await column_exists("tuning_jobs", "tuning_config"):
        logger.info("Adding 'tuning_config' column to tuning_jobs table...")
        await conn.execute(text("ALTER TABLE tuning_jobs ADD COLUMN tuning_config JSON"))
        logger.info("'tuning_config' column added!")

    # Migration: Add conversation_id to tuning_jobs (for Agent Chat integration)
    if not await column_exists("tuning_jobs", "conversation_id"):
        logger.info("Adding 'conversation_id' column to tuning_jobs table...")
        await conn.execute(text("ALTER TABLE tuning_jobs ADD COLUMN conversation_id INTEGER"))
        logger.info("'conversation_id' column added!")

    # Migration: Add os_type to workers (for Mac native deployment support)
    if not await column_exists("workers", "os_type"):
        logger.info("Adding 'os_type' column to workers table...")
        await conn.execute(
            text("ALTER TABLE workers ADD COLUMN os_type VARCHAR(50) DEFAULT 'linux'")
        )
        logger.info("'os_type' column added!")

    # Migration: Add gpu_type to workers (for Mac Apple Silicon detection)
    if not await column_exists("workers", "gpu_type"):
        logger.info("Adding 'gpu_type' column to workers table...")
        await conn.execute(
            text("ALTER TABLE workers ADD COLUMN gpu_type VARCHAR(50) DEFAULT 'nvidia'")
        )
        logger.info("'gpu_type' column added!")

    # Migration: Add capabilities to workers (for backend availability tracking)
    if not await column_exists("workers", "capabilities"):
        logger.info("Adding 'capabilities' column to workers table...")
        await conn.execute(text("ALTER TABLE workers ADD COLUMN capabilities JSON"))
        logger.info("'capabilities' column added!")

    # Migration: Add parent_app_id to apps (for monitoring services like Prometheus)
    if not await column_exists("apps", "parent_app_id"):
        logger.info("Adding 'parent_app_id' column to apps table...")
        await conn.execute(
            text("ALTER TABLE apps ADD COLUMN parent_app_id INTEGER REFERENCES apps(id)")
        )
        logger.info("'parent_app_id' column added!")


async def init_db():
    """Initialize database tables and run migrations"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Run schema migrations for any new columns
            await _run_migrations(conn)
    except Exception as e:
        # Ignore "already exists" errors from race conditions with multiple workers
        if "already exists" in str(e):
            logger.debug("Database tables already exist, skipping creation")
        else:
            raise

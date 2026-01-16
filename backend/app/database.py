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

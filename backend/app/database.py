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


def _get_column_type_sql(column) -> str:
    """Convert SQLAlchemy column type to SQLite type string."""
    from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text

    col_type = type(column.type)

    if col_type == Integer or "Integer" in str(col_type):
        return "INTEGER"
    elif col_type == String or "String" in str(col_type):
        length = getattr(column.type, "length", None)
        return f"VARCHAR({length})" if length else "VARCHAR(255)"
    elif col_type == Text or "Text" in str(col_type):
        return "TEXT"
    elif col_type == Boolean or "Boolean" in str(col_type):
        return "BOOLEAN"
    elif col_type == Float or "Float" in str(col_type):
        return "FLOAT"
    elif col_type == DateTime or "DateTime" in str(col_type):
        return "DATETIME"
    elif col_type == JSON or "JSON" in str(col_type):
        return "JSON"
    else:
        # Default fallback
        return "TEXT"


async def _run_migrations(conn):
    """Auto-detect and add missing columns by comparing models with database schema."""
    from sqlalchemy import text

    async def get_table_columns(table_name: str) -> set[str]:
        """Get all column names from a database table."""
        try:
            result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
            return {row[1] for row in result.fetchall()}
        except Exception:
            return set()

    async def table_exists(table_name: str) -> bool:
        """Check if a table exists in the database."""
        result = await conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": table_name},
        )
        return result.fetchone() is not None

    # Iterate through all tables defined in models
    for table_name, table in Base.metadata.tables.items():
        # Skip if table doesn't exist yet (will be created by create_all)
        if not await table_exists(table_name):
            continue

        # Get existing columns in database
        existing_columns = await get_table_columns(table_name)

        # Check each column in the model
        for column in table.columns:
            if column.name not in existing_columns:
                # Build ALTER TABLE statement
                col_type = _get_column_type_sql(column)

                # Handle default values
                default_clause = ""
                if column.default is not None:
                    default_val = column.default.arg
                    if callable(default_val):
                        default_val = default_val(None)
                    if isinstance(default_val, str):
                        default_clause = f" DEFAULT '{default_val}'"
                    elif isinstance(default_val, bool):
                        default_clause = f" DEFAULT {1 if default_val else 0}"
                    elif default_val is not None:
                        default_clause = f" DEFAULT {default_val}"

                sql = (
                    f"ALTER TABLE {table_name} ADD COLUMN {column.name} {col_type}{default_clause}"
                )

                logger.info(f"Auto-migration: Adding '{column.name}' column to {table_name}...")
                try:
                    await conn.execute(text(sql))
                    logger.info(f"Column '{column.name}' added to {table_name}!")
                except Exception as e:
                    logger.warning(f"Failed to add column {column.name} to {table_name}: {e}")


async def init_db():
    """Initialize database tables and run migrations"""
    # Import all models to register them with Base.metadata
    # This ensures all tables are created by create_all()
    import app.models  # noqa: F401

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

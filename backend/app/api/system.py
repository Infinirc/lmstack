"""System management API routes"""

import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.api_key import Usage

router = APIRouter()
settings = get_settings()

# Backup directory
BACKUP_DIR = Path("./backups")
BACKUP_DIR.mkdir(exist_ok=True)


class BackupInfo(BaseModel):
    """Backup file information"""

    filename: str
    size: int
    created_at: str


class BackupListResponse(BaseModel):
    """Response for listing backups"""

    items: list[BackupInfo]
    total: int


class MessageResponse(BaseModel):
    """Simple message response"""

    message: str
    details: str | None = None


@router.post("/clear-stats", response_model=MessageResponse)
async def clear_statistics(
    db: AsyncSession = Depends(get_db),
):
    """
    Clear all usage statistics data.

    This will delete all records from the usage table.
    This action cannot be undone.
    """
    try:
        # Delete all usage records
        result = await db.execute(delete(Usage))
        await db.commit()

        deleted_count = result.rowcount

        return MessageResponse(
            message="Statistics cleared successfully",
            details=f"Deleted {deleted_count} usage records",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear statistics: {str(e)}"
        )


@router.get("/backups", response_model=BackupListResponse)
async def list_backups():
    """
    List all available database backups.
    """
    backups = []

    if BACKUP_DIR.exists():
        for file in sorted(BACKUP_DIR.glob("*.db"), key=os.path.getmtime, reverse=True):
            stat = file.stat()
            backups.append(
                BackupInfo(
                    filename=file.name,
                    size=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                )
            )

    return BackupListResponse(items=backups, total=len(backups))


@router.post("/backup", response_model=MessageResponse)
async def create_backup():
    """
    Create a database backup.

    Creates a timestamped copy of the database file.
    """
    try:
        # Get database path from URL
        db_url = settings.database_url
        if "sqlite" not in db_url:
            raise HTTPException(
                status_code=400, detail="Backup only supported for SQLite databases"
            )

        # Extract database file path
        # Format: sqlite+aiosqlite:///./lmstack.db
        db_path = db_url.split("///")[-1]
        db_file = Path(db_path)

        if not db_file.exists():
            raise HTTPException(status_code=404, detail="Database file not found")

        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"lmstack_backup_{timestamp}.db"
        backup_path = BACKUP_DIR / backup_filename

        # Copy database file
        shutil.copy2(db_file, backup_path)

        return MessageResponse(
            message="Backup created successfully", details=f"Saved as {backup_filename}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create backup: {str(e)}"
        )


@router.get("/backup/{filename}")
async def download_backup(filename: str):
    """
    Download a specific backup file.
    """
    # Validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    backup_path = BACKUP_DIR / filename

    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found")

    return FileResponse(
        path=backup_path, filename=filename, media_type="application/octet-stream"
    )


@router.post("/restore/{filename}", response_model=MessageResponse)
async def restore_backup(filename: str):
    """
    Restore database from a backup file.

    WARNING: This will replace the current database with the backup.
    The server should be restarted after restore for changes to take effect.
    """
    # Validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    backup_path = BACKUP_DIR / filename

    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found")

    try:
        # Get database path
        db_url = settings.database_url
        db_path = db_url.split("///")[-1]
        db_file = Path(db_path)

        # Create a backup of current database before restore
        if db_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pre_restore_backup = BACKUP_DIR / f"pre_restore_{timestamp}.db"
            shutil.copy2(db_file, pre_restore_backup)

        # Restore from backup
        shutil.copy2(backup_path, db_file)

        return MessageResponse(
            message="Database restored successfully",
            details="Please restart the server for changes to take full effect",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to restore backup: {str(e)}"
        )


@router.delete("/backup/{filename}", response_model=MessageResponse)
async def delete_backup(filename: str):
    """
    Delete a backup file.
    """
    # Validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    backup_path = BACKUP_DIR / filename

    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found")

    try:
        backup_path.unlink()
        return MessageResponse(
            message="Backup deleted successfully", details=f"Deleted {filename}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete backup: {str(e)}"
        )


@router.post("/restore-upload", response_model=MessageResponse)
async def restore_from_upload(file: UploadFile = File(...)):
    """
    Restore database from an uploaded backup file.

    WARNING: This will replace the current database.
    The server should be restarted after restore for changes to take effect.
    """
    if not file.filename or not file.filename.endswith(".db"):
        raise HTTPException(
            status_code=400, detail="Invalid file. Please upload a .db file"
        )

    try:
        # Get database path
        db_url = settings.database_url
        db_path = db_url.split("///")[-1]
        db_file = Path(db_path)

        # Create a backup of current database before restore
        if db_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pre_restore_backup = BACKUP_DIR / f"pre_restore_{timestamp}.db"
            shutil.copy2(db_file, pre_restore_backup)

        # Save uploaded file as new database
        content = await file.read()
        with open(db_file, "wb") as f:
            f.write(content)

        return MessageResponse(
            message="Database restored from upload successfully",
            details="Please restart the server for changes to take full effect",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to restore from upload: {str(e)}"
        )

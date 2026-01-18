"""App Sync Service

Synchronizes app status with actual container state.
This is important after system reboot to ensure database status matches reality.
"""

import asyncio
import logging

import httpx
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import async_session_maker
from app.models.app import App, AppStatus

logger = logging.getLogger(__name__)


class AppSyncService:
    """Service for synchronizing app status with actual container state."""

    # Configuration
    CONTAINER_CHECK_TIMEOUT = 10  # seconds
    MAX_CONCURRENT_CHECKS = 5  # limit concurrent checks

    async def sync_all_apps(self) -> dict:
        """Synchronize all app statuses.

        Returns:
            dict with sync statistics
        """
        logger.info("Starting app status synchronization...")

        stats = {
            "total": 0,
            "running_verified": 0,
            "container_missing": 0,
            "errors": 0,
            "skipped": 0,
        }

        async with async_session_maker() as db:
            # Get all apps that should be running or are in transitional states
            result = await db.execute(
                select(App)
                .where(
                    App.status.in_(
                        [
                            AppStatus.RUNNING.value,
                            AppStatus.STARTING.value,
                            AppStatus.PULLING.value,
                        ]
                    )
                )
                .options(selectinload(App.worker))
            )
            apps = result.scalars().all()
            stats["total"] = len(apps)

            if not apps:
                logger.info("No active apps to sync")
                return stats

            logger.info(f"Found {len(apps)} active apps to check")

            # Check apps with limited concurrency
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CHECKS)

            async def check_with_semaphore(app: App):
                async with semaphore:
                    return await self._check_and_update_app(app, db)

            tasks = [check_with_semaphore(a) for a in apps]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"App check failed: {result}")
                    stats["errors"] += 1
                elif isinstance(result, str):
                    if result == "running_verified":
                        stats["running_verified"] += 1
                    elif result == "container_missing":
                        stats["container_missing"] += 1
                    elif result == "skipped":
                        stats["skipped"] += 1

            await db.commit()

        logger.info(
            f"App sync complete: {stats['running_verified']} running, "
            f"{stats['container_missing']} missing, {stats['errors']} errors"
        )

        return stats

    async def _check_and_update_app(self, app: App, db) -> str:
        """Check a single app and update its status.

        Returns:
            Status string: running_verified, container_missing, skipped
        """
        logger.debug(f"Checking app {app.id}: {app.name}")

        if not app.worker:
            logger.warning(f"App {app.id} has no worker, skipping")
            return "skipped"

        if not app.container_id:
            # If app is still being deployed (STARTING/PULLING), skip it
            if app.status in (AppStatus.STARTING.value, AppStatus.PULLING.value):
                logger.debug(f"App {app.id} is still deploying, skipping")
                return "skipped"
            # Only mark as error if app claims to be RUNNING but has no container
            logger.warning(f"App {app.id} has no container_id, marking as error")
            app.status = AppStatus.ERROR.value
            app.status_message = "Container ID missing"
            return "container_missing"

        # Check if worker is online
        if app.worker.status != "online":
            logger.warning(f"App {app.id}: worker offline")
            app.status = AppStatus.ERROR.value
            app.status_message = f"Worker {app.worker.name} is offline"
            return "container_missing"

        # Check container status via worker API
        try:
            async with httpx.AsyncClient(timeout=self.CONTAINER_CHECK_TIMEOUT) as client:
                response = await client.get(
                    f"http://{app.worker.address}/containers/{app.container_id}"
                )

                if response.status_code == 404:
                    # Container doesn't exist
                    logger.warning(f"App {app.name}: container not found")
                    app.status = AppStatus.ERROR.value
                    app.status_message = "Container not found. Please redeploy."
                    return "container_missing"

                if response.status_code == 200:
                    container_info = response.json()
                    state = container_info.get("state", "").lower()

                    if state == "running":
                        if app.status != AppStatus.RUNNING.value:
                            app.status = AppStatus.RUNNING.value
                            app.status_message = None
                        logger.debug(f"App {app.name}: running and verified")
                        return "running_verified"

                    elif state in ("exited", "dead"):
                        logger.warning(f"App {app.name}: container {state}")
                        app.status = AppStatus.STOPPED.value
                        app.status_message = f"Container {state}"
                        return "container_missing"

                    else:
                        logger.debug(f"App {app.name}: container state {state}")
                        return "running_verified"

        except httpx.ConnectError:
            logger.warning(f"App {app.id}: cannot connect to worker")
            app.status = AppStatus.ERROR.value
            app.status_message = f"Cannot connect to worker {app.worker.name}"
            return "container_missing"

        except Exception as e:
            logger.error(f"Error checking app {app.name}: {e}")
            return "skipped"

        return "skipped"


# Global instance
app_sync_service = AppSyncService()

"""Worker Sync Service

Checks all workers' online status by pinging their health endpoints.
This is important on startup to ensure database status matches reality.
"""

import asyncio
import logging

import httpx
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import async_session_maker
from app.models.app import App, AppStatus
from app.models.deployment import Deployment, DeploymentStatus
from app.models.worker import Worker, WorkerStatus

logger = logging.getLogger(__name__)


class WorkerSyncService:
    """Service for synchronizing worker status with actual state."""

    HEALTH_CHECK_TIMEOUT = 5  # seconds
    MAX_CONCURRENT_CHECKS = 10

    async def sync_all_workers(self) -> dict:
        """Check all workers' online status.

        Returns:
            dict with sync statistics
        """
        logger.info("Starting worker status synchronization...")

        stats = {
            "total": 0,
            "online": 0,
            "offline": 0,
            "errors": 0,
        }

        async with async_session_maker() as db:
            # Get all workers that are marked as online
            result = await db.execute(
                select(Worker).where(Worker.status == WorkerStatus.ONLINE.value)
            )
            workers = result.scalars().all()
            stats["total"] = len(workers)

            if not workers:
                logger.info("No online workers to check")
                return stats

            logger.info(f"Checking {len(workers)} workers...")

            # Check workers with limited concurrency
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CHECKS)

            async def check_with_semaphore(worker: Worker):
                async with semaphore:
                    return await self._check_worker(worker, db)

            tasks = [check_with_semaphore(w) for w in workers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Worker check failed: {result}")
                    stats["errors"] += 1
                elif result == "online":
                    stats["online"] += 1
                elif result == "offline":
                    stats["offline"] += 1

            await db.commit()

        logger.info(
            f"Worker sync complete: {stats['online']} online, "
            f"{stats['offline']} offline, {stats['errors']} errors"
        )

        return stats

    async def _check_worker(self, worker: Worker, db) -> str:
        """Check a single worker's status by pinging its health endpoint.

        Returns:
            "online" or "offline"
        """
        # Skip local workers that don't have a real address
        if not worker.address or worker.address.startswith("local"):
            return "online"

        try:
            async with httpx.AsyncClient(timeout=self.HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(f"http://{worker.address}/health")
                if response.status_code == 200:
                    logger.debug(f"Worker {worker.name}: online")
                    return "online"
                else:
                    logger.warning(
                        f"Worker {worker.name}: unhealthy (status {response.status_code})"
                    )
                    worker.status = WorkerStatus.OFFLINE.value
                    return "offline"

        except (httpx.ConnectError, httpx.ConnectTimeout):
            logger.warning(f"Worker {worker.name}: offline (unreachable)")
            worker.status = WorkerStatus.OFFLINE.value
            return "offline"

        except Exception as e:
            logger.error(f"Error checking worker {worker.name}: {e}")
            worker.status = WorkerStatus.OFFLINE.value
            return "offline"

    async def refresh_online_workers_resources(self) -> dict:
        """Refresh deployment and app status for all online workers.

        This should be called after worker sync to update resource status.

        Returns:
            dict with refresh statistics
        """
        logger.info("Refreshing resources on online workers...")

        stats = {
            "workers_checked": 0,
            "deployments_updated": 0,
            "apps_updated": 0,
            "errors": 0,
        }

        async with async_session_maker() as db:
            # Get all online workers
            result = await db.execute(
                select(Worker).where(Worker.status == WorkerStatus.ONLINE.value)
            )
            workers = result.scalars().all()

            if not workers:
                logger.info("No online workers to refresh")
                return stats

            stats["workers_checked"] = len(workers)
            worker_ids = [w.id for w in workers]

            # Refresh deployments on these workers
            deploy_result = await db.execute(
                select(Deployment)
                .where(
                    Deployment.worker_id.in_(worker_ids),
                    Deployment.status.in_(
                        [
                            DeploymentStatus.RUNNING.value,
                            DeploymentStatus.STARTING.value,
                        ]
                    ),
                )
                .options(selectinload(Deployment.worker))
            )
            deployments = deploy_result.scalars().all()

            for deployment in deployments:
                try:
                    updated = await self._refresh_deployment_status(deployment, db)
                    if updated:
                        stats["deployments_updated"] += 1
                except Exception as e:
                    logger.error(f"Error refreshing deployment {deployment.id}: {e}")
                    stats["errors"] += 1

            # Refresh apps on these workers
            app_result = await db.execute(
                select(App)
                .where(
                    App.worker_id.in_(worker_ids),
                    App.status.in_(
                        [
                            AppStatus.RUNNING.value,
                            AppStatus.STARTING.value,
                            AppStatus.PULLING.value,
                        ]
                    ),
                )
                .options(selectinload(App.worker))
            )
            apps = app_result.scalars().all()

            for app in apps:
                try:
                    updated = await self._refresh_app_status(app, db)
                    if updated:
                        stats["apps_updated"] += 1
                except Exception as e:
                    logger.error(f"Error refreshing app {app.id}: {e}")
                    stats["errors"] += 1

            await db.commit()

        logger.info(
            f"Resource refresh complete: {stats['deployments_updated']} deployments, "
            f"{stats['apps_updated']} apps updated, {stats['errors']} errors"
        )

        return stats

    async def _refresh_deployment_status(self, deployment: Deployment, db) -> bool:
        """Refresh a single deployment's status by checking container.

        Returns:
            True if status was updated, False otherwise
        """
        if not deployment.worker or not deployment.container_id:
            return False

        try:
            async with httpx.AsyncClient(timeout=self.HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(
                    f"http://{deployment.worker.address}/containers/{deployment.container_id}"
                )

                if response.status_code == 404:
                    # Container doesn't exist
                    if deployment.status != DeploymentStatus.ERROR.value:
                        deployment.status = DeploymentStatus.ERROR.value
                        deployment.status_message = "Container not found. Please redeploy."
                        logger.warning(f"Deployment {deployment.name}: container not found")
                        return True
                    return False

                if response.status_code == 200:
                    container_info = response.json()
                    state = container_info.get("state", "").lower()

                    if state == "running":
                        # Container is running, status is valid
                        logger.debug(f"Deployment {deployment.name}: container running")
                        return False

                    elif state in ("exited", "dead"):
                        if deployment.status != DeploymentStatus.ERROR.value:
                            deployment.status = DeploymentStatus.ERROR.value
                            deployment.status_message = f"Container {state}. Please restart."
                            logger.warning(f"Deployment {deployment.name}: container {state}")
                            return True
                        return False

        except httpx.ConnectError:
            # Worker unreachable - don't change status, worker sync handles this
            logger.debug(f"Deployment {deployment.name}: worker unreachable")
            return False
        except Exception as e:
            logger.error(f"Error checking deployment {deployment.name}: {e}")
            return False

        return False

    async def _refresh_app_status(self, app: App, db) -> bool:
        """Refresh a single app's status by checking container.

        Returns:
            True if status was updated, False otherwise
        """
        if not app.worker or not app.container_id:
            return False

        try:
            async with httpx.AsyncClient(timeout=self.HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(
                    f"http://{app.worker.address}/containers/{app.container_id}"
                )

                if response.status_code == 404:
                    # Container doesn't exist
                    if app.status != AppStatus.ERROR.value:
                        app.status = AppStatus.ERROR.value
                        app.status_message = "Container not found. Please redeploy."
                        logger.warning(f"App {app.name}: container not found")
                        return True
                    return False

                if response.status_code == 200:
                    container_info = response.json()
                    state = container_info.get("state", "").lower()

                    if state == "running":
                        # Container is running, status is valid
                        logger.debug(f"App {app.name}: container running")
                        return False

                    elif state in ("exited", "dead"):
                        if app.status != AppStatus.STOPPED.value:
                            app.status = AppStatus.STOPPED.value
                            app.status_message = f"Container {state}"
                            logger.warning(f"App {app.name}: container {state}")
                            return True
                        return False

        except httpx.ConnectError:
            # Worker unreachable - don't change status, worker sync handles this
            logger.debug(f"App {app.name}: worker unreachable")
            return False
        except Exception as e:
            logger.error(f"Error checking app {app.name}: {e}")
            return False

        return False


# Global instance
worker_sync_service = WorkerSyncService()

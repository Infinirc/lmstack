"""Worker Sync Service

Checks all workers' online status by pinging their health endpoints.
This is important on startup to ensure database status matches reality.
"""

import asyncio
import logging

import httpx
from sqlalchemy import select

from app.database import async_session_maker
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


# Global instance
worker_sync_service = WorkerSyncService()

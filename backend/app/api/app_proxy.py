"""App Proxy Routes

Proxies requests to deployed apps like Open WebUI through LMStack.
This allows users to access apps via a single LMStack URL.
"""
import logging
from typing import Optional, AsyncIterator

import httpx
from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.app import App, AppStatus

logger = logging.getLogger(__name__)
router = APIRouter()

# HTTP client settings
PROXY_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


async def get_app_by_type(app_type: str, db: AsyncSession) -> Optional[App]:
    """Get running app by type."""
    result = await db.execute(
        select(App).where(
            App.app_type == app_type,
            App.status == AppStatus.RUNNING.value,
        )
    )
    return result.scalar_one_or_none()


async def proxy_request(
    request: Request,
    target_url: str,
    path: str = "",
) -> Response:
    """Proxy a request to the target URL."""
    # Build full URL
    url = f"{target_url}{path}"
    if request.query_params:
        url = f"{url}?{request.query_params}"

    # Get request body if present
    body = await request.body()

    # Build headers for the proxied request
    headers = {}
    hop_by_hop = {
        "connection", "keep-alive", "proxy-authenticate",
        "proxy-authorization", "te", "trailers", "transfer-encoding",
        "upgrade", "host",
    }
    for key, value in request.headers.items():
        if key.lower() not in hop_by_hop:
            headers[key] = value

    # Disable compression to avoid encoding issues
    headers["Accept-Encoding"] = "identity"

    logger.info(f"Proxying {request.method} to {url}")

    try:
        async with httpx.AsyncClient(
            timeout=PROXY_TIMEOUT,
            follow_redirects=True,
        ) as client:
            response = await client.request(
                method=request.method,
                url=url,
                content=body,
                headers=headers,
            )
            logger.info(f"Proxy response: status={response.status_code}, content-type={response.headers.get('content-type')}, content-length={len(response.content)}")

            # Build response headers, excluding hop-by-hop and encoding headers
            # httpx auto-decompresses, so we must not forward Content-Encoding
            response_headers = {}
            excluded = {
                "transfer-encoding", "connection", "keep-alive",
                "content-encoding", "content-length",  # httpx decompresses, so these are invalid
            }
            for key, value in response.headers.items():
                if key.lower() not in excluded:
                    response_headers[key] = value

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

    except httpx.ConnectError as e:
        logger.error(f"Proxy connect error to {url}: {e}")
        raise HTTPException(status_code=503, detail="App is not reachable")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="App request timed out")
    except Exception as e:
        logger.exception(f"Proxy error: {e}")
        raise HTTPException(status_code=502, detail=f"Proxy error: {e}")


@router.api_route(
    "/{app_type:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_to_app(
    app_type: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Proxy requests to deployed apps.

    Routes like /apps/open-webui/* will be proxied to the Open WebUI container.
    """
    # Extract app type from path (first segment)
    path_parts = app_type.split("/", 1)
    app_type_name = path_parts[0]
    remaining_path = "/" + path_parts[1] if len(path_parts) > 1 else ""

    # Find the app
    app = await get_app_by_type(app_type_name, db)

    if not app:
        raise HTTPException(
            status_code=404,
            detail=f"App '{app_type_name}' is not deployed or not running"
        )

    # Load worker relationship
    await db.refresh(app, ["worker"])

    if not app.worker:
        raise HTTPException(status_code=500, detail="App worker not found")

    # Build target URL
    worker_host = app.worker.address.split(":")[0]
    target_url = f"http://{worker_host}:{app.port}"

    return await proxy_request(request, target_url, remaining_path)

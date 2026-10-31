import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.cache import DiskCache
from app.config import get_settings
from app.errors import SourceUnavailable
from app.providers.http import MinIntervalLimiter
from app.routers import ask, debug, kb
from app.schemas import HealthResponse

logger = logging.getLogger("agri")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    # httpx logs full request URLs at INFO, which would leak ?api-key=... into logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("API starting (env=%s, version=%s)", settings.app_env, __version__)
    cache = DiskCache(settings.cache_dir)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        app.state.http = client
        app.state.cache = cache
        app.state.nominatim_limiter = MinIntervalLimiter(1.1)  # policy: max 1 req/s
        yield
    cache.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Agri Advisor API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(SourceUnavailable)
    async def source_unavailable_handler(_: Request, exc: SourceUnavailable) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": "source_unavailable",
                "source": exc.source,
                "kind": exc.kind,
                "message": exc.user_message(),
                "retryable": exc.retryable,
                "upstream_status": exc.status_code,
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        # Liveness only. DB/Redis/upstream checks arrive with their milestones.
        return HealthResponse(
            status="ok",
            service="agri-advisor-api",
            version=__version__,
            environment=settings.app_env,
            time=datetime.now(UTC),
        )

    app.include_router(debug.router)
    app.include_router(kb.router)
    app.include_router(ask.router)
    return app


app = create_app()

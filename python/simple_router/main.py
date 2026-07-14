# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import argparse
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, Sequence

import config as config_module
import uvicorn
from database import Base, get_engine
from fastapi import FastAPI
from node import router as api_router
from sqlalchemy.exc import OperationalError

from comm_id_allocator import close_allocator, init_allocator
from generate import init_name_service


logger = logging.getLogger("simple_router.ksana_llm_simple_router")


def _configure_logging() -> None:
    """Initialise the package logger based on configuration."""

    log_format = "%(asctime)s | %(process)d | %(levelname)s | %(name)s | %(message)s"
    settings = config_module.get_settings()
    level_name = getattr(settings, "log_level", "INFO") or "INFO"
    level_value = getattr(logging, level_name.upper(), logging.INFO)

    log_dir = os.path.abspath(os.path.expanduser(settings.log_dir))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "simple_router.log")

    logging.basicConfig(
        level=level_value,
        format=log_format,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )

    package_logger = logging.getLogger("simple_router")
    package_logger.setLevel(level_value)


def _init_database() -> None:
    """Create database tables on startup if needed."""

    engine = get_engine()
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema ensured")
    except OperationalError as exc:
        message = str(exc).lower()
        if "already exists" in message:
            logger.debug("Database schema already exists; skipping creation: %s", exc)
        else:
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover - exercised at runtime
    """FastAPI lifespan hook that initialises and later tears down runtime state."""

    _configure_logging()
    _init_database()
    init_name_service()

    # Initialize comm_id allocator from settings
    settings = config_module.get_settings()
    init_allocator(
        allocator_type=settings.comm_id_allocator_type,
        redis_host=settings.redis_host,
        redis_port=settings.redis_port,
        redis_password=settings.redis_password,
        redis_db=settings.redis_db,
        redis_key=settings.comm_id_key,
    )
    
    logger.info("Prefill-decode router service started")
    yield
    
    # Cleanup
    await close_allocator()
    logger.info("Prefill-decode router service stopping")


app = FastAPI(
    title="Prefill-Decode Router",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)

__all__ = ["app", "main"]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Factory for the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Run the Prefill-Decode router service",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9080,
        help="Port to listen on (default: 9080)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of worker processes to spawn (default: 6)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the uvicorn worker group with the configured FastAPI app."""

    args = _build_arg_parser().parse_args(argv)
    _configure_logging()

    logger.info(
        "Starting Prefill-Decode router on %s:%s with %s workers",
        args.host,
        args.port,
        args.workers,
    )

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

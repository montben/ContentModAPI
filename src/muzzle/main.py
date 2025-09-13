"""
Main FastAPI application for Muzzle - AI Content Moderation API.

This module creates and configures the FastAPI application instance
with all necessary middleware, routing, and lifespan events.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging

from muzzle.core.config import settings
from muzzle.api.endpoints import text, image


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events for the application.
    This replaces the deprecated @app.on_event decorators.
    """
    # Startup tasks
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"API docs: {'/docs' if settings.enable_docs else 'disabled'}")

    # TODO: Initialize database connection
    # TODO: Load ML models
    # TODO: Initialize Redis connection

    yield  # Application runs here

    # Shutdown tasks
    logger.info(f"Shutting down {settings.app_name}")
    # TODO: Close database connections
    # TODO: Clean up ML models
    # TODO: Close Redis connections


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """

    # Create FastAPI app with configuration from settings
    app = FastAPI(
        title=settings.app_name,
        description="A robust AI-powered API that screens user-generated text and images for toxicity, hate speech, and inappropriate content.",
        version=settings.app_version,
        debug=settings.debug,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_redoc else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
    )

    # Add trusted host middleware for security
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(text.router, prefix="/api", tags=["text-moderation"])
    app.include_router(image.router, prefix="/api", tags=["image-moderation"])

    return app


# Create the FastAPI application instance
app = create_application()


@app.get("/")
async def root():
    """
    Root endpoint - health check and basic info.

    Returns:
        dict: Basic application information
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI Content Moderation Microservice",
        "status": "healthy",
        "docs_url": "/docs" if settings.enable_docs else None,
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        dict: Application health status
    """
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }




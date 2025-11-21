from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import logging

from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting up application...")
    from app.api.routes import clip_service, faiss_service
    from PIL import Image
    import numpy as np

    logger.info("Loading CLIP model...")
    clip_service.load_model()

    logger.info("Warming up CLIP model with dummy inference...")
    # Warmup: Run dummy inference to prepare model
    dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
    _ = clip_service.get_image_embedding(dummy_image)
    logger.info("Model warmup complete")

    logger.info("Loading FAISS index...")
    faiss_service.load_index()

    logger.info(f"Startup complete! Index contains {faiss_service.size} vectors")

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    description="CLIP-based image search pipeline for architectural images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add global exception handler
from app.utils.error_handlers import handle_global_exception
app.add_exception_handler(Exception, handle_global_exception)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes (import after app creation to avoid circular imports)
from app.api.routes import router
app.include_router(router, prefix="/api/v1", tags=["search"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ARCHINZA Search Pipeline API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
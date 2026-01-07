"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging

# Set up logging
logger = setup_logging()

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Production-grade RAG system API"
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["RAG"])

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the UI homepage."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "Knowledge AI API",
        "version": settings.api_version,
        "docs": "/docs",
        "ui": "/static/index.html"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI

from app.api.routes.analysis import router as analysis_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    version="1.0.0",
)

app.include_router(analysis_router, prefix="/api/v1")


@app.get("/health")
def healthcheck():
    return {"status": "ok", "app": settings.app_name}

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.domain.schemas import AnalysisResponse, StreamAnalysisRequest, VideoAnalysisRequest
from app.services.analysis_orchestrator import AnalysisOrchestrator

router = APIRouter(tags=["analysis"])


@router.post("/analysis/video", response_model=AnalysisResponse)
def analyze_video(request: VideoAnalysisRequest):
    settings = get_settings()
    orchestrator = AnalysisOrchestrator(settings)
    try:
        return orchestrator.analyze_video_from_cloud(
            video_name=request.video_name,
            run_fall_detection=request.run_fall_detection,
            run_violence_detection=request.run_violence_detection,
            pre_event_seconds=request.pre_event_seconds,
            post_event_seconds=request.post_event_seconds,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/analysis/stream", response_model=AnalysisResponse)
def analyze_stream(request: StreamAnalysisRequest):
    settings = get_settings()
    orchestrator = AnalysisOrchestrator(settings)
    try:
        return orchestrator.analyze_stream(
            stream_source=request.stream_source,
            run_fall_detection=request.run_fall_detection,
            run_violence_detection=request.run_violence_detection,
            pre_event_seconds=request.pre_event_seconds,
            post_event_seconds=request.post_event_seconds,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

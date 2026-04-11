from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Event Detection Backend"
    debug: bool = True

    temp_dir: Path = Path("./data/temp")
    output_dir: Path = Path("./data/output")

    google_drive_credentials_json: Path = Path("./credentials/service_account.json")
    google_drive_source_folder_id: str | None = None
    google_drive_event_folder_id: str | None = None
    google_drive_oauth_client_secret: str | None = None
    google_drive_oauth_token_json: str | None = None
    
    S3_BUCKET_NAME: str = "YOUR BUCKET NAME"
    S3_REGION: str = "YOUR REGION"
    AWS_ACCESS_KEY_ID: str = "YOUR ACCESS KEY ID"
    AWS_SECRET_ACCESS_KEY: str = "YOUR SECRET ACCESS KEY"

    fall_event_api_url: str = "https://8918-42-112-211-205.ngrok-free.app/PKA_ElderGuard/events"
    violence_event_api_url: str = "https://8918-42-112-211-205.ngrok-free.app/PKA_ElderGuard/events"
    event_api_timeout_seconds: float = 2.0

    default_pre_event_seconds: int = 120
    default_post_event_seconds: int = 120

    fall_model_xml: Path = Path("./models/movenet_multipose_lightning_256x256_FP32.xml")
    fall_device: str = "CPU"
    fall_score_thresh: float = 0.2
    fall_trigger_seconds: float = 2.0
    fall_min_positive_frames: int = 2
    fall_max_negative_frames: int = 8

    violence_model_pt: Path = Path("./models/best.pt")
    violence_conf: float = 0.25
    violence_imgsz: int = 416
    violence_device: str | None = None
    violence_trigger_seconds: float = 2.0
    violence_min_positive_frames: int = 2
    violence_max_negative_frames: int = 8

    default_video_fps_fallback: float = 25.0

    @property
    def event_clips_dir(self) -> Path:
        return self.output_dir / "event_clips"

    @property
    def downloaded_videos_dir(self) -> Path:
        return self.temp_dir / "videos"

    def ensure_directories(self) -> None:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.event_clips_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_videos_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
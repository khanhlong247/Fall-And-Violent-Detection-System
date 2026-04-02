# AI Backend: Fall Detection + Violent Detection

Backend FastAPI hỗ trợ:
- phân tích video theo request
- phân tích stream
- fall detection và violent detection chạy độc lập
- lưu clip sự kiện với padding trước/sau 2 phút
- upload clip sự kiện lên Google Drive
- gọi API riêng cho từng loại sự kiện

## Chạy nhanh

```bash
uv venv venv --python 3.10
venv/Scripts/activate
uv pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API

### Phân tích video từ cloud bằng `video_name`

```bash
curl -X POST http://localhost:8000/api/v1/analysis/video   -H "Content-Type: application/json"   -d '{
    "video_name": "camera_01_2026_04_02.mp4",
    "run_fall_detection": true,
    "run_violence_detection": true
  }'
```

### Phân tích stream

```bash
curl -X POST http://localhost:8000/api/v1/analysis/stream   -H "Content-Type: application/json"   -d '{
    "stream_source": "rtsp://user:pass@camera/stream",
    "run_fall_detection": true,
    "run_violence_detection": true
  }'
```

## Biến môi trường mẫu

```env
APP_NAME=AI Event Detection Backend
DEBUG=true
TEMP_DIR=./data/temp
OUTPUT_DIR=./data/output
GOOGLE_DRIVE_CREDENTIALS_JSON=./credentials/service_account.json
GOOGLE_DRIVE_SOURCE_FOLDER_ID=
GOOGLE_DRIVE_EVENT_FOLDER_ID=
FALL_EVENT_API_URL=http://localhost:9001/fall-events
VIOLENCE_EVENT_API_URL=http://localhost:9002/violence-events
EVENT_API_TIMEOUT_SECONDS=2.0
DEFAULT_PRE_EVENT_SECONDS=120
DEFAULT_POST_EVENT_SECONDS=120
FALL_MODEL_XML=./models/movenet_multipose_lightning_256x256_FP32.xml
FALL_DEVICE=CPU
VIOLENCE_MODEL_PT=./models/best.pt
VIOLENCE_CONF=0.25
```

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from app.utils.file_utils import ensure_parent

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


class GoogleDriveStorage:
    def __init__(self, credentials_json: Path, token_json: Path | None = None):
        credentials_json = Path(credentials_json)
        if not credentials_json.exists():
            raise FileNotFoundError(f"Google OAuth client secret not found: {credentials_json}")

        self.credentials_json = credentials_json
        self.token_json = Path(token_json) if token_json else credentials_json.parent / "token.json"

        credentials = self._build_oauth_credentials()
        self.service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    def _build_oauth_credentials(self) -> Credentials:
        credentials = None

        if self.token_json.exists():
            credentials = Credentials.from_authorized_user_file(str(self.token_json), DRIVE_SCOPES)

        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_json),
                    DRIVE_SCOPES,
                )
                credentials = flow.run_local_server(port=0)

            ensure_parent(self.token_json)
            self.token_json.write_text(credentials.to_json(), encoding="utf-8")

        return credentials

    def find_file_by_name(self, video_name: str, folder_id: str | None = None) -> dict | None:
        name_escaped = video_name.replace("'", "\\'")
        query = f"name = '{name_escaped}' and trashed = false"
        if folder_id:
            query += f" and '{folder_id}' in parents"

        response = self.service.files().list(
            q=query,
            fields="files(id, name, mimeType, webViewLink)",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files = response.get("files", [])
        return files[0] if files else None

    def download_file_by_name(self, video_name: str, destination_path: Path, folder_id: str | None = None) -> Path:
        file_meta = self.find_file_by_name(video_name, folder_id=folder_id)
        if not file_meta:
            raise FileNotFoundError(f"Video not found on Google Drive: {video_name}")

        request = self.service.files().get_media(fileId=file_meta["id"])
        ensure_parent(destination_path)

        with destination_path.open("wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        return destination_path

    def upload_file(self, local_path: Path, remote_name: str, folder_id: str | None = None) -> dict:
        metadata = {"name": remote_name}
        if folder_id:
            metadata["parents"] = [folder_id]

        media = MediaFileUpload(str(local_path), resumable=True)
        created = self.service.files().create(
            body=metadata,
            media_body=media,
            fields="id, name, webViewLink",
            supportsAllDrives=True,
        ).execute()
        return created
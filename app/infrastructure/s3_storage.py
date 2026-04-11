import boto3
from pathlib import Path
from botocore.exceptions import ClientError
from app.domain.entities import StoredClip
from app.domain.enums import EventType

class S3StorageService:
    def __init__(self, bucket_name: str, region: str, access_key: str, secret_key: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

    def find_file_by_name(self, video_name: str, folder_id: str | None = None) -> dict | None:
        """
        Tìm file trên S3. 
        Trong S3, folder_id thường là prefix (ví dụ: 'raw_videos/')
        """
        prefix = folder_id if folder_id else ""
        try:
            # S3 Key chính xác thường là prefix + video_name
            full_key = f"{prefix}{video_name}" if prefix else video_name
            
            # Kiểm tra xem object có tồn tại không
            self.s3_client.head_object(Bucket=self.bucket_name, Key=full_key)
            
            # Trả về dict có 'id' để khớp với logic cũ của Google Drive
            return {"id": full_key, "name": video_name}
        except ClientError:
            # Nếu không tìm thấy bằng key trực tiếp, thử list để tìm
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            for obj in response.get('Contents', []):
                if video_name in obj['Key']:
                    return {"id": obj['Key'], "name": video_name}
            return None

    def download_file_by_name(self, video_name: str, destination_path: Path, folder_id: str | None = None) -> Path:
        """
        Tải file từ S3 về máy local.
        """
        file_meta = self.find_file_by_name(video_name, folder_id=folder_id)
        if not file_meta:
            raise FileNotFoundError(f"Video not found on S3: {video_name}")

        s3_key = file_meta["id"]
        # Đảm bảo thư mục cha tồn tại
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[DEBUG][S3] Downloading {s3_key} to {destination_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, str(destination_path))
            return destination_path
        except Exception as e:
            print(f"[ERROR][S3] Download failed: {e}")
            raise e

    def upload_clip(self, clip) -> StoredClip:
        """
        Logic upload clip sau khi xử lý xong (giữ nguyên của bạn)
        """
        file_path = Path(clip.local_path)
        
        # Khớp với folder trên S3 của bạn: fall_event/ hoặc violent_event/
        folder = "fall_event" if clip.event_type == EventType.FALL else "violent_event"
        object_name = f"{folder}/{file_path.name}"
        
        try:
            self.s3_client.upload_file(
                str(file_path), 
                self.bucket_name, 
                object_name,
                ExtraArgs={
                    'ContentType': 'video/mp4',
                    'ContentDisposition': 'inline'
                }
            )
            
            remote_link = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_name},
                ExpiresIn=604800
            )

            return StoredClip(
                event_type=clip.event_type,
                local_path=clip.local_path,
                remote_file_id=object_name,
                remote_link=remote_link,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                fps=clip.fps
            )
        except Exception as e:
            print(f"[ERROR][S3] Upload failed: {e}")
            raise e
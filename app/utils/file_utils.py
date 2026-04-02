from pathlib import Path
from uuid import uuid4


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_unique_filename(prefix: str, suffix: str) -> str:
    return f"{prefix}_{uuid4().hex}{suffix}"


def normalize_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").strip()

"""Download the model checkpoint configured through environment variables."""

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    APP_ROOT / "model_artifacts" / "medicalnet_resnet18_top50.pth"
)


def configured_model_path() -> Path:
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
    return model_path if model_path.is_absolute() else APP_ROOT / model_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    model_path = configured_model_path()
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return 0

    model_url = os.getenv("MODEL_URL", "").strip()
    if not model_url:
        print("MODEL_URL is required when the checkpoint is absent.", file=sys.stderr)
        return 1

    if urlparse(model_url).scheme not in {"https", "http"}:
        print("MODEL_URL must use HTTP or HTTPS.", file=sys.stderr)
        return 1

    expected_hash = os.getenv("MODEL_SHA256", "").strip().lower()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            print(f"Downloading model to {temp_path}")
            with urllib.request.urlopen(model_url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    temp_file.write(chunk)

        actual_hash = sha256(temp_path)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch: expected {expected_hash}, received {actual_hash}"
            )

        temp_path.replace(model_path)
        print(f"Model saved to {model_path}")
        print(f"SHA-256: {actual_hash}")
        return 0
    except Exception as exc:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

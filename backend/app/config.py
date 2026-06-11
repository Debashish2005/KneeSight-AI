import os
from pathlib import Path

from dotenv import load_dotenv


APP_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(APP_ROOT / "backend" / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    configured = Path(os.getenv(name, str(default))).expanduser()
    return configured if configured.is_absolute() else APP_ROOT / configured


def _csv_from_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in os.getenv(name, default).split(",")
        if value.strip()
    )


MODEL_PATH = _path_from_env(
    "MODEL_PATH",
    APP_ROOT / "model_artifacts" / "medicalnet_resnet18_top50.pth",
)
MODEL_URL = os.getenv("MODEL_URL", "").strip()
MODEL_SHA256 = os.getenv("MODEL_SHA256", "").strip().lower()

EXPECTED_SHAPE = (128, 128, 64)
TOP_K_SLICES = 50
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(16 * 1024 * 1024)))
MODEL_VERSION = os.getenv("MODEL_VERSION", "medicalnet-resnet18-top50-v1")

# Replace the default with BEST_THRESHOLD_I3D from validation. ROC-AUC is
# threshold-independent, but the displayed Normal/Abnormal class is not.
CLASSIFICATION_THRESHOLD = float(
    os.getenv("CLASSIFICATION_THRESHOLD", "0.50")
)
THRESHOLD_SOURCE = os.getenv("THRESHOLD_SOURCE", "temporary-default")
CORS_ORIGINS = _csv_from_env(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)

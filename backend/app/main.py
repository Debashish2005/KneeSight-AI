from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import (
    CORS_ORIGINS,
    EXPECTED_SHAPE,
    MAX_UPLOAD_BYTES,
    MODEL_VERSION,
    THRESHOLD_SOURCE,
)
from app.services.inference import ScanValidationError, inference_service

app = FastAPI(
    title="KneeSight AI API",
    description="Research API for knee MRI abnormality screening.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_version: str
    expected_shape: tuple[int, int, int]
    device: str
    threshold_source: str
    model_error: str | None = None


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    threshold: float
    model_version: str
    disclaimer: str


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_ready=inference_service.ready,
        model_version=MODEL_VERSION,
        expected_shape=EXPECTED_SHAPE,
        device=str(inference_service.device),
        threshold_source=THRESHOLD_SOURCE,
        model_error=inference_service.load_error,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(scan: UploadFile = File(...)) -> PredictionResponse:
    if not scan.filename or not scan.filename.lower().endswith(".npy"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .npy MRI volumes are supported in this version.",
        )

    payload = await scan.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The uploaded volume exceeds the 16 MB limit.",
        )

    try:
        volume = np.load(BytesIO(payload), allow_pickle=False)
        result = inference_service.predict(volume)
    except (ValueError, OSError, EOFError, ScanValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return PredictionResponse(
        prediction=result.prediction,
        probability=result.probability,
        threshold=result.threshold,
        model_version=result.model_version,
        disclaimer=(
            "Probability refers to the abnormal class. Research use only; "
            "this output is not a medical diagnosis."
        ),
    )

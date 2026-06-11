from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_checkpoint_state() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert isinstance(response.json()["model_ready"], bool)
    assert response.json()["expected_shape"] == [128, 128, 64]


def test_rejects_wrong_volume_shape() -> None:
    payload = BytesIO()
    np.save(payload, np.zeros((16, 16, 16), dtype=np.float32))

    response = client.post(
        "/api/predict",
        files={"scan": ("invalid.npy", payload.getvalue(), "application/octet-stream")},
    )

    assert response.status_code == 422
    assert "Expected MRI shape" in response.json()["detail"]


def test_rejects_malformed_numpy_file() -> None:
    response = client.post(
        "/api/predict",
        files={"scan": ("broken.npy", b"not a numpy array", "application/octet-stream")},
    )

    assert response.status_code == 422


def test_preprocessing_matches_model_input_shape() -> None:
    from app.services.inference import inference_service

    volume = np.arange(128 * 128 * 64, dtype=np.float32).reshape(128, 128, 64)
    tensor = inference_service.preprocess(volume)

    assert tuple(tensor.shape) == (1, 1, 50, 128, 128)
    assert abs(float(tensor.mean())) < 1e-5
    assert abs(float(tensor.std(unbiased=False)) - 1.0) < 1e-5

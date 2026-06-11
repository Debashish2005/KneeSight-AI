from io import BytesIO
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import SimpleITK as sitk
from fastapi.testclient import TestClient

from app.main import app
from app.services.ingestion import ScanIngestionError, ingest_scan

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


def test_converts_nifti_to_model_geometry() -> None:
    array = np.random.default_rng(7).normal(size=(20, 30, 12)).astype(np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 0.8, 1.5))

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as nifti_file:
        nifti_path = Path(nifti_file.name)
    try:
        sitk.WriteImage(image, str(nifti_path))
        ingested = ingest_scan("knee.nii.gz", nifti_path.read_bytes())
    finally:
        nifti_path.unlink(missing_ok=True)

    assert ingested.source_format == "NIfTI"
    assert ingested.original_shape == (20, 30, 12)
    assert ingested.volume.shape == (128, 128, 64)
    assert np.isfinite(ingested.volume).all()
    assert ingested.warnings


def test_converts_zipped_dess_dicom_series() -> None:
    payload = _synthetic_dicom_zip("SAG 3D DESS")
    ingested = ingest_scan("dess-series.zip", payload)

    assert ingested.source_format == "DICOM ZIP"
    assert ingested.original_shape == (10, 24, 20)
    assert ingested.volume.shape == (128, 128, 64)
    assert any("SAG 3D DESS" in warning for warning in ingested.warnings)


def test_rejects_non_dess_dicom_series() -> None:
    payload = _synthetic_dicom_zip("SAG T1")

    try:
        ingest_scan("t1-series.zip", payload)
    except ScanIngestionError as exc:
        assert "does not identify a 3D DESS series" in str(exc)
    else:
        raise AssertionError("A non-DESS DICOM series should be rejected.")


def test_rejects_unsafe_zip_paths() -> None:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("../outside.dcm", b"unsafe")

    try:
        ingest_scan("unsafe.zip", payload.getvalue())
    except ScanIngestionError as exc:
        assert "unsafe file path" in str(exc)
    else:
        raise AssertionError("A ZIP path traversal entry should be rejected.")


def _synthetic_dicom_zip(description: str) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        array = (
            np.random.default_rng(11).random((10, 24, 20)) * 1000
        ).astype(np.uint16)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.7, 0.7, 1.2))

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        writer.SetImageIO("GDCMImageIO")
        series_uid = "1.2.826.0.1.3680043.2.1125.20260611"

        for slice_index in range(image.GetDepth()):
            image_slice = image[:, :, slice_index]
            image_slice.SetMetaData("0008|0060", "MR")
            image_slice.SetMetaData("0008|103e", description)
            image_slice.SetMetaData("0018|1030", description)
            image_slice.SetMetaData("0020|000e", series_uid)
            image_slice.SetMetaData("0020|0013", str(slice_index + 1))
            position = image.TransformIndexToPhysicalPoint((0, 0, slice_index))
            image_slice.SetMetaData(
                "0020|0032",
                "\\".join(str(value) for value in position),
            )
            image_slice.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")
            writer.SetFileName(str(root / f"{slice_index:03d}.dcm"))
            writer.Execute(image_slice)

        payload = BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
            for dicom_file in sorted(root.glob("*.dcm")):
                archive.write(dicom_file, dicom_file.name)
        return payload.getvalue()

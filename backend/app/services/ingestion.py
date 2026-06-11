from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import numpy as np
import SimpleITK as sitk

from app.config import (
    EXPECTED_SHAPE,
    MAX_DICOM_FILES,
    MAX_EXTRACTED_BYTES,
    REQUIRE_DESS_DICOM,
)


SUPPORTED_SUFFIXES = (".npy", ".nii", ".nii.gz", ".zip")
DESS_KEYWORDS = ("dess", "dual echo steady state")


class ScanIngestionError(ValueError):
    """Raised when an uploaded scan cannot be converted safely."""


@dataclass(frozen=True)
class IngestedVolume:
    volume: np.ndarray
    source_format: str
    original_shape: tuple[int, int, int]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DicomSeries:
    files: tuple[str, ...]
    description: str
    is_dess: bool


def supported_filename(filename: str) -> bool:
    lowered = filename.lower()
    return any(lowered.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def ingest_scan(filename: str, payload: bytes) -> IngestedVolume:
    lowered = filename.lower()
    if lowered.endswith(".npy"):
        return _load_numpy(payload)
    if lowered.endswith((".nii", ".nii.gz")):
        suffix = ".nii.gz" if lowered.endswith(".nii.gz") else ".nii"
        return _load_nifti(payload, suffix)
    if lowered.endswith(".zip"):
        return _load_dicom_zip(payload)
    raise ScanIngestionError(
        "Supported formats are .npy, .nii, .nii.gz, and zipped DICOM series."
    )


def _load_numpy(payload: bytes) -> IngestedVolume:
    try:
        volume = np.load(BytesIO(payload), allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise ScanIngestionError(f"Invalid NumPy volume: {exc}") from exc

    if volume.ndim != 3:
        raise ScanIngestionError(
            f"Expected a 3D NumPy volume, received shape {volume.shape}."
        )
    return IngestedVolume(
        volume=volume,
        source_format="NumPy",
        original_shape=tuple(int(value) for value in volume.shape),
    )


def _load_nifti(payload: bytes, suffix: str) -> IngestedVolume:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as scan_file:
            temporary_path = Path(scan_file.name)
            scan_file.write(payload)

        assert temporary_path is not None
        try:
            image = sitk.ReadImage(str(temporary_path))
        except RuntimeError as exc:
            raise ScanIngestionError(f"Invalid NIfTI volume: {exc}") from exc
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    return _convert_medical_image(
        image,
        source_format="NIfTI",
        warnings=(
            "NIfTI does not reliably identify the MRI sequence. Only sagittal "
            "3D DESS knee MRI should be used with this model.",
            "Raw-format conversion is experimental and was not part of the "
            "reported 0.8251 test AUC evaluation.",
        ),
    )


def _load_dicom_zip(payload: bytes) -> IngestedVolume:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        _extract_zip_safely(payload, root)
        series = _find_dicom_series(root)

        if not series:
            raise ScanIngestionError(
                "No readable DICOM MRI series was found in the ZIP archive."
            )

        selected = max(
            series,
            key=lambda candidate: (candidate.is_dess, len(candidate.files)),
        )
        if REQUIRE_DESS_DICOM and not selected.is_dess:
            raise ScanIngestionError(
                "The DICOM metadata does not identify a 3D DESS series. "
                "This model was trained only for sagittal 3D DESS knee MRI."
            )

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(selected.files)
        try:
            image = reader.Execute()
        except RuntimeError as exc:
            raise ScanIngestionError(
                f"The selected DICOM series could not be decoded: {exc}"
            ) from exc

    warnings = [
        "DICOM files were processed in a temporary directory and deleted "
        "after conversion.",
        "Raw-format conversion is experimental and was not part of the "
        "reported 0.8251 test AUC evaluation.",
    ]
    if selected.description:
        warnings.append(f"Selected DICOM series: {selected.description}.")
    return _convert_medical_image(
        image,
        source_format="DICOM ZIP",
        warnings=tuple(warnings),
    )


def _extract_zip_safely(payload: bytes, destination: Path) -> None:
    try:
        archive = zipfile.ZipFile(BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise ScanIngestionError("The uploaded file is not a valid ZIP archive.") from exc

    with archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if not members:
            raise ScanIngestionError("The DICOM ZIP archive is empty.")
        if len(members) > MAX_DICOM_FILES:
            raise ScanIngestionError(
                f"The archive contains more than {MAX_DICOM_FILES} files."
            )

        extracted_size = sum(member.file_size for member in members)
        if extracted_size > MAX_EXTRACTED_BYTES:
            raise ScanIngestionError(
                "The uncompressed DICOM archive exceeds the configured limit."
            )

        for member in members:
            relative_path = PurePosixPath(member.filename.replace("\\", "/"))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ScanIngestionError(
                    "The ZIP archive contains an unsafe file path."
                )

            target = destination.joinpath(*relative_path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _find_dicom_series(root: Path) -> list[DicomSeries]:
    candidates: list[DicomSeries] = []
    for directory in [root, *(path for path in root.rglob("*") if path.is_dir())]:
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory))
        except RuntimeError:
            continue

        for series_id in series_ids or ():
            files = tuple(
                sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                    str(directory),
                    series_id,
                )
            )
            if len(files) < 8:
                continue

            metadata = _read_dicom_metadata(files[0])
            if metadata.get("0008|0060", "").upper() != "MR":
                continue

            description = " ".join(
                value
                for value in (
                    metadata.get("0008|103e", ""),
                    metadata.get("0018|1030", ""),
                )
                if value
            ).strip()
            lowered_description = description.lower()
            candidates.append(
                DicomSeries(
                    files=files,
                    description=description,
                    is_dess=any(
                        keyword in lowered_description for keyword in DESS_KEYWORDS
                    ),
                )
            )
    return candidates


def _read_dicom_metadata(filename: str) -> dict[str, str]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()
    try:
        reader.ReadImageInformation()
    except RuntimeError:
        return {}

    return {
        key: reader.GetMetaData(key).strip()
        for key in ("0008|0060", "0008|103e", "0018|1030")
        if reader.HasMetaDataKey(key)
    }


def _convert_medical_image(
    image: sitk.Image,
    source_format: str,
    warnings: tuple[str, ...],
) -> IngestedVolume:
    if image.GetDimension() != 3:
        raise ScanIngestionError(
            f"Expected one 3D MRI volume, received {image.GetDimension()} dimensions."
        )
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ScanIngestionError("Multi-channel medical images are not supported.")

    original_array_shape = tuple(
        int(value) for value in sitk.GetArrayViewFromImage(image).shape
    )
    try:
        oriented = sitk.DICOMOrient(image, "LPS")
    except RuntimeError as exc:
        raise ScanIngestionError(
            f"The scan orientation could not be standardized: {exc}"
        ) from exc

    # SimpleITK uses x, y, z while NumPy returns z, y, x. Setting the physical
    # x axis to 64 makes the final NumPy array 128 x 128 x 64, with sagittal
    # slices along the last axis as expected by the training notebook.
    output_size = (EXPECTED_SHAPE[2], EXPECTED_SHAPE[1], EXPECTED_SHAPE[0])
    input_size = oriented.GetSize()
    input_spacing = oriented.GetSpacing()
    output_spacing = tuple(
        (
            input_spacing[index] * max(input_size[index] - 1, 1)
            / max(output_size[index] - 1, 1)
        )
        for index in range(3)
    )

    resampled = sitk.Resample(
        oriented,
        output_size,
        sitk.Transform(),
        sitk.sitkLinear,
        oriented.GetOrigin(),
        output_spacing,
        oriented.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )
    volume = sitk.GetArrayFromImage(resampled).astype(np.float32, copy=False)

    if volume.shape != EXPECTED_SHAPE:
        raise ScanIngestionError(
            f"Conversion produced shape {volume.shape}, expected {EXPECTED_SHAPE}."
        )
    if not np.isfinite(volume).all():
        raise ScanIngestionError("The converted MRI contains invalid values.")
    if float(volume.std()) < 1e-8:
        raise ScanIngestionError("The converted MRI has no usable intensity variation.")

    return IngestedVolume(
        volume=volume,
        source_format=source_format,
        original_shape=original_array_shape,
        warnings=warnings,
    )

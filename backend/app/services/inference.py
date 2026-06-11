from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.config import (
    CLASSIFICATION_THRESHOLD,
    EXPECTED_SHAPE,
    MODEL_PATH,
    MODEL_VERSION,
    TOP_K_SLICES,
)


class ScanValidationError(ValueError):
    """Raised when an uploaded MRI array does not match the trained pipeline."""


@dataclass(frozen=True)
class Prediction:
    prediction: str
    probability: float
    threshold: float
    model_version: str


class MedicalNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class MedicalNetResNet18Classifier(nn.Module):
    """Architecture used by the final Top-50 Kaggle notebook."""

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=1, dilation=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=1, dilation=4)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

        layers = [
            MedicalNetBasicBlock(
                self.in_channels,
                out_channels,
                stride,
                dilation,
                downsample,
            )
        ]
        self.in_channels = out_channels
        layers.extend(
            MedicalNetBasicBlock(
                out_channels,
                out_channels,
                dilation=dilation,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(x).squeeze(1)


class MedicalNetInferenceService:
    """Owns checkpoint loading and the exact notebook inference pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: MedicalNetResNet18Classifier | None = None
        self.load_error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.load_error = f"Checkpoint not found at {self.model_path}."
            return

        try:
            model = MedicalNetResNet18Classifier(dropout=0.5)
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=True,
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self.model = model.to(self.device)
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = f"Could not load MedicalNet checkpoint: {exc}"

    @property
    def ready(self) -> bool:
        return self.model is not None

    def validate_volume(self, volume: np.ndarray) -> np.ndarray:
        if volume.shape != EXPECTED_SHAPE:
            raise ScanValidationError(
                f"Expected MRI shape {EXPECTED_SHAPE}, received {volume.shape}."
            )
        if not np.issubdtype(volume.dtype, np.number):
            raise ScanValidationError("MRI volume must contain numeric values.")
        if not np.isfinite(volume).all():
            raise ScanValidationError("MRI volume contains NaN or infinite values.")
        return volume.astype(np.float32, copy=False)

    def preprocess(self, volume: np.ndarray) -> torch.Tensor:
        volume = self.validate_volume(volume)

        slice_scores = volume.std(axis=(0, 1))
        selected_slices = np.argsort(slice_scores)[-TOP_K_SLICES:]
        selected_slices = np.sort(selected_slices)
        selected_volume = volume[:, :, selected_slices]

        mean = float(selected_volume.mean())
        std = float(selected_volume.std())
        selected_volume = (selected_volume - mean) / (std + 1e-8)

        # 128 x 128 x 50 -> batch x channel x depth x height x width
        selected_volume = np.transpose(selected_volume, (2, 0, 1))
        tensor = torch.from_numpy(
            np.ascontiguousarray(selected_volume, dtype=np.float32)
        )
        return tensor.unsqueeze(0).unsqueeze(0)

    def predict(self, volume: np.ndarray) -> Prediction:
        if not self.ready:
            raise RuntimeError(
                self.load_error or "The trained MedicalNet checkpoint is unavailable."
            )

        tensor = self.preprocess(volume).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            abnormal_probability = float(torch.sigmoid(logits).item())

        prediction = (
            "Abnormal"
            if abnormal_probability >= CLASSIFICATION_THRESHOLD
            else "Normal"
        )
        return Prediction(
            prediction=prediction,
            probability=abnormal_probability,
            threshold=CLASSIFICATION_THRESHOLD,
            model_version=MODEL_VERSION,
        )


inference_service = MedicalNetInferenceService()

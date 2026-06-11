# KneeSight AI

KneeSight AI is a full-stack research application for knee MRI abnormality
screening. A React interface uploads a preprocessed 3D MRI volume to a FastAPI
service, where a pretrained MedicalNet ResNet-18 model returns a Normal or
Abnormal prediction with its probability.

> Research and educational use only. This project is not a medical device and
> must not be used to diagnose a patient or recommend treatment.

## Result

| Model | Input strategy | Independent test ROC-AUC |
| --- | --- | ---: |
| MedicalNet ResNet-18 | Top 50 intensity-selected slices | **0.8251** |

The model was evaluated on an independent test set of 447 OAI knee MRI
volumes. ROC-AUC measures ranking performance and does not depend on the
classification threshold. The displayed Normal/Abnormal label currently uses
a temporary threshold of `0.50` until the validation-selected threshold is
exported from the final notebook.

## Features

- Responsive React and Vite healthcare interface
- Animated Normal/Abnormal probability visualization
- Result-aware, non-prescriptive follow-up guidance
- FastAPI inference and health endpoints
- `.npy`, NIfTI, and zipped DICOM ingestion
- Physical orientation standardization and 3D resampling with SimpleITK
- DICOM archive safety, MR modality, and DESS sequence checks
- Pretrained 3D MedicalNet ResNet-18 with strict checkpoint loading
- Training-consistent Top-50 slice selection and per-volume normalization
- Environment-based frontend URL, CORS, model path, and threshold settings
- Dockerfiles and Docker Compose for reproducible startup
- Automated API and preprocessing tests

## Architecture

```text
React browser client
       |
       | multipart/form-data
       v
FastAPI validation and preprocessing
       |
       | tensor: 1 x 1 x 50 x 128 x 128
       v
MedicalNet ResNet-18
       |
       v
Prediction, abnormal-class probability, threshold, model metadata
```

See [Architecture](docs/ARCHITECTURE.md), [API](docs/API.md), and
[Deployment](docs/DEPLOYMENT.md) for the full walkthrough. Read
[Input Formats](docs/INPUT_FORMATS.md) before testing raw scans and
[Safety](docs/SAFETY.md) before changing the result guidance.

## Project Structure

```text
medical-mri-app/
  backend/
    app/                 FastAPI application and PyTorch inference
    scripts/             Model download helper
    tests/               API and preprocessing tests
  frontend/
    public/              Static assets
    src/                 React application and styles
  model_artifacts/       Local checkpoint location; weights are Git-ignored
  docs/                  Architecture, API, and deployment guides
  docker-compose.yml
```

## Local Setup

### 1. Add the checkpoint

Place the trained checkpoint at:

```text
model_artifacts/medicalnet_resnet18_top50.pth
```

The checkpoint is about 127 MB and is intentionally excluded from GitHub.

### 2. Start the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`, with interactive Swagger
documentation at `http://127.0.0.1:8000/docs`.

### 3. Start the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Docker

With Docker Desktop running and the checkpoint in `model_artifacts/`:

```powershell
docker compose up --build
```

Open `http://localhost:5173`.

## Tests

```powershell
cd backend
pip install -r requirements-dev.txt
python -m pytest
```

```powershell
cd frontend
npm run build
```

## Input Contract

The current version accepts one de-identified scan:

- `.npy`: prepared array with exact shape `(128, 128, 64)`
- `.nii` or `.nii.gz`: one 3D NIfTI knee MRI volume
- `.zip`: one DICOM MRI series; metadata must identify 3D DESS
- Values: finite numeric MRI intensities
- Maximum upload: 256 MB by default

NIfTI and DICOM volumes are reoriented to LPS coordinates and resampled to the
model geometry. Uploaded files are processed in memory or a temporary
directory and are not deliberately persisted by the API.

The reported `0.8251` test ROC-AUC applies to prepared OAI arrays, not arbitrary
raw NIfTI or DICOM scans. Raw conversion is therefore marked experimental in
the API and interface. Different scanner protocols, fields of view, sequences,
and populations can substantially change model behavior.

## Model Weights and Data

The source code is released under the MIT License. The MedicalNet pretrained
weights, trained checkpoint, and OAI dataset have their own terms and are not
granted by this repository's license. Verify those terms before redistribution
or commercial use.

## Research Context

This application extends an ongoing May-June 2026 research internship project
at the National Institute of Technology Warangal under the guidance of
Prof. V Rama, Department of Electronics and Communication Engineering.

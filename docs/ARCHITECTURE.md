# Architecture

## Request Flow

1. The React client checks `GET /api/health` and displays whether the model is
   ready.
2. The user chooses a `.npy` MRI volume.
3. The browser sends the scan to `POST /api/predict` as multipart form data.
4. FastAPI checks the extension and upload size.
5. The ingestion service either loads an exact-shape NumPy array or converts
   NIfTI/DICOM using SimpleITK.
6. Raw medical images are reoriented to LPS physical coordinates and resampled
   to `128 x 128 x 64`. DICOM ZIPs also receive archive safety, modality, and
   DESS metadata checks.
7. The inference service validates shape, data type, and finite values.
8. The service ranks the 64 slices by intensity standard deviation and keeps
   the 50 highest-scoring slices in their original spatial order.
9. The selected volume is standardized and converted to a
   `1 x 1 x 50 x 128 x 128` PyTorch tensor.
10. MedicalNet ResNet-18 produces one logit. A sigmoid converts it to the
   abnormal-class probability.
11. The API compares that probability with the configured threshold and returns
   the result.
12. React animates the probability gauge and displays both class probabilities.

## Frontend

The frontend uses React 19, Vite, Lucide icons, and handwritten responsive CSS.
During development, Vite proxies `/api` to `http://127.0.0.1:8000`. In a hosted
build, `VITE_API_BASE_URL` points directly to the deployed backend.

Important files:

- `frontend/src/App.jsx`: state, upload flow, API calls, and page sections
- `frontend/src/styles.css`: responsive layout and visual design
- `frontend/vite.config.js`: local development server and API proxy

## Backend

FastAPI owns HTTP validation and response models. The inference service owns
the PyTorch architecture, checkpoint loading, volume preprocessing, and
prediction.

Important files:

- `backend/app/main.py`: API routes, request limits, CORS, and errors
- `backend/app/config.py`: environment-driven runtime configuration
- `backend/app/services/ingestion.py`: NIfTI/DICOM conversion and archive safety
- `backend/app/services/inference.py`: MedicalNet and preprocessing pipeline
- `backend/scripts/download_model.py`: optional deployment-time checkpoint fetch

## Why the Checkpoint Is Separate

The trained checkpoint is about 127 MB, which exceeds GitHub's ordinary
100 MB file limit. It is also a derived research artifact that may have
different redistribution terms from the source code. The repository therefore
tracks model metadata and download tooling, not the weights themselves.

## Current Boundaries

- The API processes one volume per request.
- There is no authentication, persistent database, patient record, or audit log.
- The class threshold is still the temporary `0.50` default.
- Raw NIfTI/DICOM conversion was not included in the reported model evaluation.
- The output is for research evaluation, not clinical decision-making.

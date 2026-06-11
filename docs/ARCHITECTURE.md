# Architecture

## Request Flow

1. The React client checks `GET /api/health` and displays whether the model is
   ready.
2. The user chooses a `.npy` MRI volume.
3. The browser sends the volume to `POST /api/predict` as multipart form data.
4. FastAPI checks the extension and upload size, then NumPy loads the array
   with pickling disabled.
5. The inference service validates shape, data type, and finite values.
6. The service ranks the 64 slices by intensity standard deviation and keeps
   the 50 highest-scoring slices in their original spatial order.
7. The selected volume is standardized and converted to a
   `1 x 1 x 50 x 128 x 128` PyTorch tensor.
8. MedicalNet ResNet-18 produces one logit. A sigmoid converts it to the
   abnormal-class probability.
9. The API compares that probability with the configured threshold and returns
   the result.
10. React animates the probability gauge and displays both class probabilities.

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
- `backend/app/services/inference.py`: MedicalNet and preprocessing pipeline
- `backend/scripts/download_model.py`: optional deployment-time checkpoint fetch

## Why the Checkpoint Is Separate

The trained checkpoint is about 127 MB, which exceeds GitHub's ordinary
100 MB file limit. It is also a derived research artifact that may have
different redistribution terms from the source code. The repository therefore
tracks model metadata and download tooling, not the weights themselves.

## Current Boundaries

- Only preprocessed `.npy` volumes are accepted.
- The API processes one volume per request.
- There is no authentication, persistent database, patient record, or audit log.
- The class threshold is still the temporary `0.50` default.
- The output is for research evaluation, not clinical decision-making.

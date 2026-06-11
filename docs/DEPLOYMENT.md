# Deployment Guide

## Recommended Layout

Deploy the two services separately:

- Frontend: static Vite build on Vercel, Netlify, Cloudflare Pages, or similar
- Backend: Docker-capable compute with enough RAM for PyTorch and the 3D model
- Checkpoint: private model registry or object storage with a stable download URL

The backend is substantially heavier than a normal CRUD API. Confirm available
memory and cold-start limits before selecting a hosting plan.

## Backend Environment

| Variable | Purpose |
| --- | --- |
| `MODEL_PATH` | Local checkpoint destination |
| `MODEL_URL` | Deployment-time checkpoint download URL |
| `MODEL_SHA256` | Optional integrity check for the downloaded checkpoint |
| `MODEL_VERSION` | Version returned by API responses |
| `CLASSIFICATION_THRESHOLD` | Normal/Abnormal decision threshold |
| `THRESHOLD_SOURCE` | Human-readable threshold provenance |
| `CORS_ORIGINS` | Comma-separated frontend origins |
| `MAX_UPLOAD_BYTES` | Maximum request file size |
| `PORT` | Hosted server port |

For a hosted backend, set `MODEL_PATH` to a writable location such as
`/tmp/models/medicalnet_resnet18_top50.pth` if the platform filesystem is
read-only outside `/tmp`.

The Docker startup command downloads the checkpoint when it is absent and then
starts Uvicorn:

```text
python scripts/download_model.py &&
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Frontend Environment

Set the backend's public URL at build time:

```text
VITE_API_BASE_URL=https://your-backend.example.com
```

Do not include a trailing slash. Then add the frontend's exact public origin to
the backend:

```text
CORS_ORIGINS=https://your-frontend.example.com
```

## Deployment Checklist

1. Upload the checkpoint to approved storage.
2. Calculate and configure its SHA-256 checksum.
3. Deploy the backend Dockerfile with the model environment variables.
4. Confirm `/api/health` reports `model_ready: true`.
5. Deploy the frontend with `VITE_API_BASE_URL`.
6. Set `CORS_ORIGINS` to the final frontend URL and redeploy the backend.
7. Test a known valid Normal and Abnormal research volume.
8. Confirm no uploaded file or response is being logged by the hosting layer.
9. Keep the research-only disclaimer visible.
10. Do not upload identifiable patient data.

## Production Gaps

Public demonstration hosting is reasonable for de-identified sample data, but
this prototype is not ready for clinical use. A clinical system would require
security review, authentication, encryption and retention controls, monitoring,
dataset shift evaluation, calibration, explainability review, regulatory work,
and validation by qualified medical professionals.

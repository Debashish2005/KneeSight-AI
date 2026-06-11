# API Guide

The local base URL is `http://127.0.0.1:8000`.

## Health

```http
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "model_ready": true,
  "model_version": "medicalnet-resnet18-top50-v1",
  "expected_shape": [128, 128, 64],
  "device": "cpu",
  "threshold_source": "temporary-default",
  "model_error": null
}
```

The service can be online while `model_ready` is `false`. Check `model_error`
when diagnosing a missing or incompatible checkpoint.

## Predict

```http
POST /api/predict
Content-Type: multipart/form-data
```

The form field must be named `scan`.

PowerShell example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/predict `
  -F "scan=@C:\path\to\volume.npy"
```

Python example:

```python
import requests

with open("volume.npy", "rb") as scan:
    response = requests.post(
        "http://127.0.0.1:8000/api/predict",
        files={"scan": ("volume.npy", scan, "application/octet-stream")},
        timeout=120,
    )

response.raise_for_status()
print(response.json())
```

Example response:

```json
{
  "prediction": "Normal",
  "probability": 0.061,
  "threshold": 0.5,
  "model_version": "medicalnet-resnet18-top50-v1",
  "disclaimer": "Probability refers to the abnormal class. Research use only; this output is not a medical diagnosis."
}
```

`probability` always refers to the Abnormal class, even when the returned
prediction is Normal.

## Error Responses

| Status | Meaning |
| --- | --- |
| `413` | File exceeds the configured upload limit |
| `415` | File is not a `.npy` volume |
| `422` | Invalid NumPy file, wrong shape, or invalid values |
| `503` | Model checkpoint is missing or could not be loaded |

## PUT and DELETE

This API currently performs stateless inference, so it has no resource that
needs updating or deleting. `PUT` and `DELETE` would make sense after adding
stored cases, for example:

```python
@app.put("/api/cases/{case_id}")
def update_case(case_id: int, payload: CaseUpdate):
    ...


@app.delete("/api/cases/{case_id}", status_code=204)
def delete_case(case_id: int):
    ...
```

Do not add these routes until the application has authentication, a database,
clear retention rules, and authorization checks.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model_artifacts/medicalnet_resnet18_top50.pth

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch==2.6.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["sh", "-c", "python scripts/download_model.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# Dockerfile for the regression training + FastAPI inference service.
#
# CPU base image: builds and runs fast, and works on your Mac. To actually
# use a GPU, deploy on a Linux host with an NVIDIA GPU + nvidia-container-
# toolkit and swap this base for a CUDA image (see README / compose comments).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models

WORKDIR /app

# Dependency layer is cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Default: serve the API. Training is a separate one-off command, e.g.
#   docker compose run --rm api python train.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

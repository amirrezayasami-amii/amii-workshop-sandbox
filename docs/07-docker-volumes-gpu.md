# Module 07 — Volumes (Live Reload) & GPU Passthrough

[← FastAPI](06-fastapi-inference.md) · [Index](README.md) · [Next: CI/CD Pipeline →](08-cicd-pipeline.md)

## Learning objectives

- Use a **bind-mount volume** so code changes sync into the container instantly.
- Enable **live reload** so the server restarts on edit.
- Configure **GPU passthrough** for NVIDIA hardware — and understand exactly
  when it does and doesn't work.
- Understand the CPU-vs-CUDA image size trade-off.

## Prerequisites

- Modules 05–06 complete; the API runs in Docker.

---

## 1. The problem volumes solve

By default, a Docker image is a **snapshot**. Edit `app.py` on your host and the
running container still runs the *old* copy baked in at build time — you'd have
to rebuild for every change. That's a miserable dev loop.

A **bind-mount volume** maps a host directory into the container so they share
the same files live.

---

## 2. The compose configuration

```yaml
services:
  api:
    build: .
    image: regression-api
    ports:
      - "8000:8000"
    volumes:
      - .:/app          # <-- host project dir mounted into the container
    environment:
      - MODEL_DIR=/app/models
    command: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Two cooperating pieces make live reload work:

1. **`volumes: - .:/app`** — the host project directory *becomes* `/app` inside
   the container. Your edits are visible immediately; no rebuild.
2. **`--reload`** — uvicorn watches the mounted files and restarts the server
   whenever one changes.

`ports: - "8000:8000"` maps container port 8000 to host port 8000 so you can
reach the API at `http://localhost:8000`.

### Bonus: the volume also carries the model out

Because the whole project is mounted, when `train.py` writes to `/app/models`
inside the container, those files land in `./models` on your host — and the
running API (same mount) serves them. **One mount gives you both live code
reload and model persistence.**

### See it live

```bash
docker compose up                 # start the API
# ...edit app.py (e.g. change the /health response)...
# uvicorn logs: "Detected change... Reloading..."
curl http://localhost:8000/health # reflects your edit — no rebuild
```

> **Windows/macOS note:** bind-mount file-change events can be slightly delayed
> on non-Linux hosts due to the virtualization layer. Usually fine; if reload
> misses events, that's the cause.

---

## 3. GPU passthrough

Deep learning is much faster on a GPU. Docker can expose a host NVIDIA GPU to a
container — with three requirements that **must all** be met:

```yaml
    # In docker-compose.yml, under the service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**Requirements:**

1. A **Linux host** with an NVIDIA GPU.
2. The **NVIDIA driver** installed on the host.
3. The **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**
   (`nvidia-container-toolkit`) installed and configured.

And the image must contain **CUDA-enabled** libraries — swap the Dockerfile base
to a CUDA build (e.g. a `pytorch/pytorch` CUDA tag, or `nvidia/cuda` + pip
torch).

Because `model.py` uses `select_device()`, **no application code changes** —
`torch.cuda.is_available()` returns `True` inside a properly configured GPU
container, and everything moves to the GPU automatically.

Verify inside a GPU container:

```bash
docker compose run --rm api python -c "import torch; print(torch.cuda.is_available())"
# True   (on a correctly configured Linux GPU host)
```

### ⚠️ Pitfall — GPU passthrough does NOT work on Docker Desktop for Mac

This workshop was developed on macOS, which is worth calling out plainly:

- **Docker Desktop for Mac has no NVIDIA runtime.** Apple Silicon / Intel Macs
  don't expose an NVIDIA GPU to Linux containers. The `deploy.devices` block
  will error (`could not select device driver "nvidia"`) if enabled there.
- That's why in this repo the GPU block is **committed but commented out**, with
  instructions — so the project runs on any machine, and you enable the block on
  a real Linux GPU host (a cloud VM, a workstation, CI with GPUs).

**Recommended teaching setup for the GPU section:** a cloud GPU VM (AWS
`g4dn`/`g5`, GCP with a T4, etc.) with the NVIDIA Container Toolkit
pre-installed, or demo it as "here's the config; here's why it's off locally."

---

## 4. Image size: CPU vs CUDA torch

Check the image size:

```bash
docker image ls regression-api
# regression-api   latest   8.49GB
```

**8.49 GB** — almost entirely PyTorch. The default `torch` wheel bundles CUDA
libraries even on a CPU-only base image, where they can't be used.

**To slim it (≈1 GB)** when you don't need GPU, install the CPU-only build:

```dockerfile
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt
```

Trade-off summary:

| Target | Base image | torch build | Approx size |
| ------ | ---------- | ----------- | ----------- |
| Local dev / CPU deploy | `python:3.12-slim` | CPU-only wheel | ~1 GB |
| GPU deploy | CUDA base (`pytorch/pytorch`) | CUDA wheel | multi-GB |

Choose per deployment target. This repo defaults to the portable CPU base with
the GPU path documented.

---

## Checkpoint ✅

- [ ] Editing `app.py` while `docker compose up` is running reloads the server
      without a rebuild.
- [ ] Training inside the container writes `models/` to your host.
- [ ] You can explain the three requirements for GPU passthrough and why it's
      off on macOS.

## Exercises

1. Add a second bind mount that maps only `./models` (read-only) and discuss
   when you'd prefer a **named volume** over a bind mount.
2. Modify the Dockerfile to install CPU-only torch and compare the new image
   size with `docker image ls`.
3. (If you have access) spin up a cloud GPU VM, enable the `deploy.devices`
   block, and confirm `torch.cuda.is_available()` is `True`.

[← FastAPI](06-fastapi-inference.md) · [Index](README.md) · [Next: CI/CD Pipeline →](08-cicd-pipeline.md)

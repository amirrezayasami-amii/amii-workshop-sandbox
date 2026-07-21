# Module 09 — Troubleshooting & Gotchas (Appendix)

[← CI/CD Pipeline](08-cicd-pipeline.md) · [Index](README.md)

A standalone reference of **every real problem we hit** while building this
project, with the symptom, the cause, and the fix. These make excellent
"predict what goes wrong" discussion prompts in a workshop.

---

## 1. `error: externally-managed-environment`

**Symptom**

```
error: externally-managed-environment
× This environment is externally managed
```

**Cause** — [PEP 668](https://peps.python.org/pep-0668/). You ran `pip install`
against an OS-managed Python (Homebrew/Debian) with no virtual environment.

**Fix** — use a virtual environment; do **not** reach for
`--break-system-packages`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. `torch` won't install on Python 3.14

**Symptom** — `pip install torch` fails or tries to build from source forever;
`No module named 'numpy'` on a fresh 3.14 interpreter.

**Cause** — the newest interpreter often lacks prebuilt wheels for scientific/ML
packages.

**Fix** — standardize on **Python 3.12** for anything involving torch. We keep:
CI matrix at 3.11–3.13, Docker image at 3.12. The API tests
`pytest.importorskip("torch")` so they *skip* (not fail) on a torch-less local
3.14 venv and still run in Docker.

---

## 3. `ModuleNotFoundError: No module named 'preprocessing'` under pytest

**Symptom** — bare `pytest` (in Docker/CI) can't import your top-level module,
even though `python -m pytest` works locally.

**Cause** — pytest's default "prepend" import mode adds the *test file's*
directory (`tests/`) to `sys.path`, not the project root. `python -m pytest`
accidentally masks this by adding the cwd.

**Fix** — `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

> This one bug appeared in **three** places (local bare pytest, Docker, and
> would-be CI). One central config fixes all of them — a good lesson in fixing
> root causes over symptoms.

---

## 4. `RecursionError` in a mock's `side_effect`

**Symptom**

```
RecursionError: maximum recursion depth exceeded
```

...from a test whose fake `read_csv` called `pd.read_csv`.

**Cause** — the `side_effect` referenced the very symbol being patched, so it
invoked the mock (itself) endlessly.

**Fix** — capture the original *before* patching:

```python
real_read_csv = pd.read_csv
with patch("preprocessing.pd.read_csv",
           side_effect=lambda p: real_read_csv(io.StringIO(text))):
    ...
```

---

## 5. GPU passthrough fails on macOS

**Symptom**

```
could not select device driver "nvidia" with capabilities: [[gpu]]
```

**Cause** — Docker Desktop for Mac has no NVIDIA runtime; Macs don't pass an
NVIDIA GPU to Linux containers.

**Fix** — keep the `deploy.devices` block **commented** locally; enable it only
on a Linux host with an NVIDIA GPU + `nvidia-container-toolkit`. `select_device()`
falls back to CPU automatically, so the app still runs everywhere.

---

## 6. The image is 8.5 GB

**Symptom** — `docker image ls` shows a multi-GB image.

**Cause** — the default `torch` wheel bundles CUDA libraries even on a CPU base.

**Fix** — for CPU deployments, install the CPU-only build:

```dockerfile
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Drops the image to roughly ~1 GB. Use the CUDA base only when actually
targeting a GPU host.

---

## 7. Docker layer cache "won't update" my code

**Symptom** — you changed code but the container runs the old version, or a
rebuild seems to skip your change.

**Causes & fixes**

- **Running an old image** — rebuild: `docker compose build`. Or use the volume
  mount + `--reload` (Module 07) so you don't rebuild at all during dev.
- **Wrong layer order** — if you `COPY . .` *before* installing requirements,
  every code edit busts the dependency cache (slow) rather than speeding things
  up. Copy `requirements.txt` and install *first*, code last.

---

## 8. Model loads but `/predict` returns 503

**Symptom** — `GET /health` shows `"model_loaded": false`; `POST /predict`
returns `503 Model not loaded`.

**Cause** — no artifacts in `MODEL_DIR`. You haven't trained yet, or the volume
isn't mounted so the container can't see `models/`.

**Fix**

```bash
docker compose run --rm api python train.py   # create models/
docker compose up                             # serve; health now true
```

Confirm the mount: `models/` should exist on the host after training.

---

## 9. CI is slow / downloads gigabytes of CUDA

**Symptom** — the `Install dependencies` step in CI takes many minutes and pulls
a huge `torch` wheel.

**Cause** — the default `torch` wheel bundles CUDA even though CI runners are
CPU-only.

**Fix** — install the CPU-only wheel *before* `requirements.txt`, so the generic
`torch` line is already satisfied:

```yaml
- run: |
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
```

---

## 10. `denied: permission_denied` / `installation not allowed` pushing to GHCR

**Symptom** — the Docker publish step fails to push to `ghcr.io`.

**Causes & fixes**

- **Missing permissions** — the job needs `permissions: packages: write`. Add it
  to the workflow (we do).
- **Uppercase in the image name** — GHCR requires **lowercase** repository names,
  but `github.repository` can contain uppercase. Lowercase it first:

  ```yaml
  - id: img
    run: echo "name=ghcr.io/${GITHUB_REPOSITORY,,}" >> "$GITHUB_OUTPUT"
  ```

- **First push visibility** — after the first successful push, the package is
  *private* by default. Make it public (or grant repo access) under the
  package's settings if others need to pull it.

---

## 11. Coverage shows 0% for `train.py`

**Symptom** — the coverage report lists `train.py` at 0%.

**Cause** — `train.py` is a CLI entrypoint that downloads a dataset; it's
exercised by running `python train.py`, not by unit tests. That's expected.

**Fix (optional)** — scope coverage to the app modules in `pyproject.toml` so the
report isn't misleading:

```toml
[tool.coverage.run]
omit = ["tests/*", "train.py", "postprocessing1.py", "preprocessing2.py"]
```

---

## Quick command reference

```bash
# Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tests
pytest -v                                   # all tests
pytest -k normalize                         # subset by name
docker compose run --rm api pytest -v       # tests inside the container
docker compose run --rm api pytest --cov=.  # with coverage (54 tests)

# Train & serve
docker compose build
docker compose run --rm api python train.py --epochs 30
docker compose up                           # API on :8000

# Inspect
docker image ls regression-api
curl -s http://localhost:8000/health
```

[← CI/CD Pipeline](08-cicd-pipeline.md) · [Index](README.md)

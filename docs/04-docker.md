# Module 04 — Docker Fundamentals

[← CI](03-ci-github-actions.md) · [Index](README.md) · [Next: The Regression Model →](05-regression-model.md)

## Learning objectives

- Explain what a container is and how it differs from a virtual environment.
- Write a `Dockerfile` and understand **layer caching**.
- Trim the build context with `.dockerignore`.
- Orchestrate builds and runs with **docker-compose**.
- Run your test suite *inside* the container.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
  and running.
- Modules 01–03 complete.

---

## 1. Why Docker?

A virtual environment isolates **Python packages**. A container isolates the
**entire runtime** — OS libraries, the Python interpreter, system tools, and
your code — into a portable image that runs identically on your laptop, a
teammate's machine, CI, and production.

> venv answers "which Python packages?" · Docker answers "which *everything*?"

---

## 2. Your first Dockerfile

A `Dockerfile` is a recipe. Each instruction creates a cached **layer**.

```dockerfile
# Use a slim Python base with reliable wheels. 3.12 avoids the 3.14 wheel gap.
FROM python:3.12-slim

# Cleaner container behavior: unbuffered logs, no .pyc files, no pip cache.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy ONLY requirements first, so this layer caches unless deps change.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Now copy the rest of the source.
COPY . .

# Default command when the container starts.
CMD ["pytest", "-v"]
```

### The single most important idea: layer caching order

Docker caches each layer and only rebuilds a layer (and everything after it) if
its inputs changed. We deliberately:

1. `COPY requirements.txt` **and install** — *before* copying the code.
2. `COPY . .` — the code, last.

Why? Your source changes constantly but your dependencies rarely do. With this
order, editing `app.py` reuses the cached (slow) dependency-install layer and
only re-runs the fast `COPY . .`. Reverse the order and every code edit
reinstalls everything.

---

## 3. `.dockerignore` — keep the build context lean

Everything in the build directory is sent to the Docker daemon as "context".
Exclude what the image doesn't need:

```gitignore
.git/
.github/
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
Dockerfile
.dockerignore
docker-compose.yml
.DS_Store
```

Excluding `.venv/` matters a lot — shipping a host virtualenv into a Linux
image is both wasteful and broken (wrong platform binaries).

---

## 4. Build and run

```bash
docker build -t my-app .
docker run --rm my-app
```

- `-t my-app` names ("tags") the image.
- `--rm` removes the container after it exits (no clutter).

**Expected:** the image builds, then `pytest -v` runs inside it.

### ⚠️ The import bug that Docker exposed

The first in-container test run failed:

```
ModuleNotFoundError: No module named 'preprocessing'
```

This is the **exact** pytest import-path issue from Module 02. It stayed hidden
locally because we'd been running `python -m pytest` (which adds the cwd to
`sys.path`), but the container's `CMD` runs the bare `pytest` entrypoint, which
does not.

**The fix is the same central one** — `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

> **Why this is a *good* thing:** the container caught a latent bug that would
> also have broken CI (which likewise runs bare `pytest`). Containers surfacing
> environment assumptions is a feature, not a bug.

Rebuild and the suite passes:

```
============================== 36 passed in 0.10s ==============================
```

---

## 5. docker-compose — stop typing long commands

`docker run` commands get long fast (ports, volumes, env vars). **Compose**
captures them in `docker-compose.yml`:

```yaml
services:
  app:
    build: .
    image: my-app
    volumes:
      - .:/app          # mount the project (more on this in Module 07)
    working_dir: /app
```

Now:

```bash
docker compose build            # build the image
docker compose run --rm app     # run the default command (pytest)
docker compose run --rm app python -c "import preprocessing; print('ok')"
```

`docker compose run` overrides the command with anything after the service name
— handy for one-off tasks like training (Module 05).

---

## Checkpoint ✅

- [ ] `Dockerfile`, `.dockerignore`, and `docker-compose.yml` exist.
- [ ] `docker compose build` succeeds.
- [ ] `docker compose run --rm app` runs the tests **inside** the container and
      they pass.

## Exercises

1. Edit a source file, rebuild, and watch which layers are `CACHED` vs. rebuilt.
   Then edit `requirements.txt` and rebuild — notice the dependency layer now
   re-runs.
2. Add a healthcheck or a second service to the compose file.
3. Run `docker image ls` and note your image size. (We revisit image size in
   Module 07 — the default `torch` makes it multi-GB.)

[← CI](03-ci-github-actions.md) · [Index](README.md) · [Next: The Regression Model →](05-regression-model.md)

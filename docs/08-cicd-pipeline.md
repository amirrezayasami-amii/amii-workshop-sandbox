# Module 08 — CI/CD Pipeline: Test the ML App & Publish the Image

[← Volumes & GPU](07-docker-volumes-gpu.md) · [Index](README.md) · [Next: Troubleshooting →](09-troubleshooting.md)

## Learning objectives

- Upgrade the minimal CI from Module 03 into a **production-grade pipeline** that
  tests the full stack (preprocessing + model + API).
- Measure **test coverage** and keep the report meaningful.
- Install **CPU-only torch** so CI is fast and cheap.
- Add a **Continuous Delivery** workflow that builds the Docker image, tests it,
  and **publishes to a container registry**.
- Understand the difference between **CI** (test every change) and **CD**
  (ship artifacts automatically).

## Prerequisites

- Modules 03–07 complete. You have the ML model, the API, tests for both, and a
  working Docker image.

---

## 1. Why the Module 03 workflow is no longer enough

Back in Module 03 the project only had preprocessing tests and three
dependencies. Now it has:

- **torch** — a large dependency whose default wheel drags in CUDA.
- **model and API tests** that need torch + fastapi installed.
- a **Docker image** worth publishing so others can `docker pull` and run it.

So we split the pipeline into two workflows with distinct jobs:

| Workflow | File | Trigger | Purpose (CI vs CD) |
| -------- | ---- | ------- | ------------------ |
| **CI** | `.github/workflows/ci.yml` | every push & PR | run the full test suite with coverage |
| **Docker** | `.github/workflows/docker-publish.yml` | push to main / tags | build, test, and publish the image |

> **CI** protects correctness on every change. **CD** turns a green build into a
> shippable artifact. Keeping them in separate files keeps each one focused.

---

## 2. The CI workflow (`ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [main, master, regression-api]
  pull_request:
  workflow_dispatch:            # manual "Run workflow" button

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true      # cancel superseded runs on the same branch

jobs:
  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          # CPU-only torch first, so the generic `torch` in requirements is
          # already satisfied and the giant CUDA wheel is never downloaded.
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt

      - name: Run test suite with coverage
        run: pytest -v --cov=. --cov-report=term-missing --cov-report=xml

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: coverage-${{ matrix.python-version }}
          path: coverage.xml
          if-no-files-found: ignore
```

### What changed from Module 03, and why

| Addition | Why it matters |
| -------- | -------------- |
| **CPU-only torch index** | The default wheel is multi-GB (CUDA). The `--index-url .../cpu` wheel is a fraction of the size, so installs are minutes faster and cheaper. Because it's installed *before* `requirements.txt`, pip sees `torch` as already satisfied. |
| **`--cov` (coverage)** | Reports which lines your tests actually execute. `term-missing` prints uncovered lines; `xml` produces a machine-readable report for upload / services like Codecov. |
| **`concurrency` + `cancel-in-progress`** | If you push twice quickly, the first (now-stale) run is cancelled — saves CI minutes. |
| **`workflow_dispatch`** | Adds a "Run workflow" button in the Actions tab for on-demand runs. |
| **`upload-artifact`** | Saves `coverage.xml` so you can download it from the run summary. `if: always()` uploads even when tests fail. |
| **Matrix trimmed to 3.11–3.12** | torch wheels are reliable here; we dropped 3.13 to keep the matrix lean (add it back if you need the coverage). |

### Keeping the coverage report honest

Run locally (in Docker) to see what CI sees:

```bash
docker compose run --rm api pytest --cov=. --cov-report=term-missing
```

```
Name               Stmts   Miss  Cover
------------------------------------------
app.py                46      0   100%
model.py              56      0   100%
preprocessing.py      51      0   100%
------------------------------------------
TOTAL                153      0   100%
54 passed
```

By default coverage also lists `train.py` at **0%** — it's a CLI entrypoint that
downloads a dataset, exercised by running it, not by unit tests. Rather than let
that skew the number, we scope coverage in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["."]
omit = [
    "tests/*",
    "train.py",            # CLI entrypoint, run manually
    "postprocessing1.py",  # empty placeholder
    "preprocessing2.py",   # empty placeholder
]

[tool.coverage.report]
show_missing = true
```

> **Teaching point:** coverage is a *tool*, not a target. Omitting an integration
> entrypoint is honest; gaming the number by testing trivial code is not.

---

## 3. The CD workflow (`docker-publish.yml`)

This builds the image, runs the tests **inside** it (so we publish exactly what
we tested), and pushes to the **GitHub Container Registry (GHCR)**.

```yaml
name: Docker

on:
  push:
    branches: [main, master]
    tags: ["v*"]
  pull_request:
  workflow_dispatch:

jobs:
  build-test-publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write            # required to push to GHCR

    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3

      # GHCR requires lowercase names; repo names may contain uppercase.
      - id: img
        run: echo "name=ghcr.io/${GITHUB_REPOSITORY,,}" >> "$GITHUB_OUTPUT"

      - name: Build image (load locally for testing)
        uses: docker/build-push-action@v6
        with:
          context: .
          load: true
          tags: ${{ steps.img.outputs.name }}:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run tests inside the image
        run: docker run --rm ${{ steps.img.outputs.name }}:ci pytest -v

      # --- publish (skipped on PRs) --------------------------------------
      - name: Log in to GHCR
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - id: meta
        if: github.event_name != 'pull_request'
        uses: docker/metadata-action@v5
        with:
          images: ${{ steps.img.outputs.name }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha
            type=ref,event=tag

      - name: Build and push
        if: github.event_name != 'pull_request'
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### The key ideas

- **Test the artifact you ship.** We `load` the built image and run `pytest`
  inside it *before* pushing. A green unit-test suite on the host isn't the same
  as the image actually working.
- **PRs build & test but never publish.** The `if: github.event_name !=
  'pull_request'` guards ensure only merges to `main`/`master` (and version
  tags) publish. PRs still get the safety of a build+test.
- **`GITHUB_TOKEN` needs `packages: write`.** No extra secret to configure —
  GitHub injects the token; you just grant the permission.
- **Lowercase the image name.** `${GITHUB_REPOSITORY,,}` (bash lowercase
  expansion) avoids GHCR rejecting names with uppercase letters.
- **Build cache (`type=gha`).** Reuses layers across runs so rebuilds are fast.
- **Semantic tags** via `metadata-action`: `latest` on the default branch, a
  `sha-<commit>` tag for traceability, and the git tag for releases (`v1.2.3`).

### Pulling the published image

After a successful run on `main`:

```bash
docker pull ghcr.io/<owner>/<repo>:latest
docker run --rm -p 8000:8000 ghcr.io/<owner>/<repo>:latest
```

> **First-push visibility:** GHCR packages are **private** by default. Make the
> package public (or grant access) in its settings if others need to pull it.

---

## 4. The full picture

```
        ┌──────────────── every push / PR ────────────────┐
        │                                                  │
   ci.yml (CI)                                  docker-publish.yml (CD)
   ├─ 3.11 ─┐                                   ├─ build image (cached)
   ├─ 3.12 ─┤ install CPU torch + deps          ├─ run pytest INSIDE image
   │        └ pytest --cov  ──▶ coverage.xml    └─ push to GHCR  (main/tags only)
   └────────────────────────────────────────────────────────────┘
```

- **On a PR:** both workflows run; nothing is published. A red check blocks the
  merge.
- **On merge to `main`:** tests run *and* the image is published to GHCR as
  `:latest` and `:sha-<commit>`.
- **On a `v*` tag:** a versioned image is published for that release.

---

## Checkpoint ✅

- [ ] `.github/workflows/ci.yml` runs the full suite with coverage on 3.11/3.12.
- [ ] `.github/workflows/docker-publish.yml` builds, tests, and (on main)
      publishes the image.
- [ ] Coverage is scoped in `pyproject.toml` and reads 100% on app modules.
- [ ] You can explain why PRs build-and-test but don't publish.

## Exercises

1. Add a **lint** job to `ci.yml` (`pip install ruff && ruff check .`) that must
   pass before tests.
2. Add a coverage **threshold**: `pytest --cov=. --cov-fail-under=90` so the
   build fails if coverage drops.
3. Change the CD target from GHCR to **Docker Hub** (swap the registry, login
   action inputs, and add `DOCKERHUB_TOKEN` as a repo secret).
4. Add a job that only runs on `v*` tags and creates a **GitHub Release** with
   the image reference in the notes.

[← Volumes & GPU](07-docker-volumes-gpu.md) · [Index](README.md) · [Next: Troubleshooting →](09-troubleshooting.md)

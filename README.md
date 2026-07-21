# Amii Engineering Workshop — Git & GitHub · Unit Testing · CI/CD

A hands-on **sandbox** repository for the Amii Engineering & Performance workshop.
It is shaped like a small but real machine-learning project — a PyTorch
regression model served through a FastAPI API and packaged with Docker — and it
exists to teach **three engineering pillars** on top of that project:

1. **Git & GitHub** — branching, pull requests, merge conflicts, rebasing, undo.
2. **Unit testing** — pytest fixtures, parametrization, mocking, coverage.
3. **CI/CD with GitHub Actions** — run the suite on every push/PR, then build,
   test, and publish the container image.

> This is a **throwaway** repo. Break things freely, reset hard, force-push your
> own feature branches — that is exactly what it is for.

The ML code is the *substrate*: something realistic to test, to run in CI, and
to ship. You do **not** need any ML background to follow the workshop.

---

## The three pillars

| Pillar | You practise… | Sandbox surface |
| ------ | ------------- | --------------- |
| **Git & GitHub** | branch → commit → PR → merge; resolve a conflict; rebase; recover with reflog | seeded branches + `config.py` conflict bait + `PARTICIPANTS.md` |
| **Unit testing** | fixtures, parametrization, mocks, coverage, red→green | `tests/` (`test_preprocessing.py`, `test_io_mocks.py`, `test_model.py`, `test_api.py`) |
| **CI/CD** | matrix test runs, coverage gates, build & publish image | `.github/workflows/ci.yml`, `.github/workflows/docker-publish.yml` |

Full teaching material is in [`docs/`](docs/README.md) — start there.

---

## Repository layout

| Path | What it is |
| ---- | ---------- |
| `config.py` | Hyperparameters — single source of truth (**and conflict bait**). |
| `preprocessing.py` | pandas/numpy data transforms — the main unit-test target. |
| `model.py` | PyTorch MLP + artifact save/load helpers. |
| `train.py` | Training script (reads `config.py`, writes `models/`). |
| `app.py` | FastAPI inference service. |
| `tests/` | The full pytest suite (fixtures, parametrization, mocks, API tests). |
| `Dockerfile`, `docker-compose.yml` | Container image + orchestration (volumes, GPU). |
| `.github/workflows/` | `ci.yml` (test matrix + coverage) and `docker-publish.yml` (build/publish). |
| `docs/` | The workshop modules. |
| `PARTICIPANTS.md` | Add yourself here during the live lab. |

---

## Branching model

| Branch | Meaning |
| ------ | ------- |
| `main` | Stable, always-green. Tagged releases live here (`v0.1.0` = the workshop baseline). |
| `staging` | The next release — features accumulate and stabilise here; CI runs on it. |
| `feature/*` | Your work — branch off `staging`, open a PR back into `staging`. |

---

## Planted exercises

The seeded history contains deliberate situations to practise on:

1. **Open a PR** — `feature/standardize-columns` is a clean feature branch ready
   to review and merge into `staging`.
2. **Resolve a conflict** — `feature/tune-lr` edits the same `LEARNING_RATE`
   line in `config.py` that `staging` already changed. Merge it and fix the
   conflict by hand.
3. **Rebase safely** — `feature/faster-epochs` was branched before `staging`
   moved ahead. Rebase it onto `staging`, then push with `--force-with-lease`.
4. **Red → green (testing)** — `feature/clip-outliers` adds a new
   `clip_outliers` function **with a failing test**. Make the test pass, then
   watch CI go green on your PR.
5. **Recover work** — practise `git reset --hard HEAD~1`, then get the commit
   back with `git reflog`.

See [`docs/git-github.md`](docs/git-github.md) for step-by-step guidance.

---

## Quick-start

```bash
# 1. Environment (Python 3.11 or 3.12 — see docs/00 for the torch/3.14 pitfall)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Confirm the suite is green
pytest -v

# 3. Start your lab branch
git switch staging
git switch -c feature/<your-name>
#   ...add your line to PARTICIPANTS.md...
git add PARTICIPANTS.md
git commit -m "feat: add <your-name> to participants"
git push -u origin feature/<your-name>       # then open a PR into staging
```

---

## Commit style — Conventional Commits

```
<type>: <short summary in the imperative mood>
```

Types: `feat`, `fix`, `docs`, `refactor`, `style`, `test`, `chore`, `ci`.
Example: `test: add parametrized cases for normalize()`

## Handy alias

```bash
git config --global alias.graph "log --oneline --graph --decorate --all"
git graph
```

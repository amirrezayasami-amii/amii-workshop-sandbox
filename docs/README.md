# Workshop: Git & GitHub · Unit Testing · CI/CD

A hands-on workshop built around **three engineering pillars** that every
collaborator on a shared codebase needs. We teach them on top of a small but
real project — a PyTorch regression model served with FastAPI and packaged in
Docker — so the skills land in a realistic setting instead of a toy repo.

> **No ML background required.** The model is just something concrete to test,
> to run in CI, and to ship. Focus on the workflow, not the math.

Every module states its objectives, gives copy-pasteable commands, shows the
**expected output**, ends with a **checkpoint**, and calls out real mistakes
that are easy to make.

---

## Who this is for

- Researchers and developers who want a practical, shared **team workflow**:
  version control, tests they can trust, and automation that catches mistakes.
- Facilitators who want a ready-to-teach curriculum with a seeded sandbox.

**Assumed knowledge:** basic Python and comfort in a terminal.

---

## The three pillars

### Pillar 1 — Git & GitHub
Branching, pull requests, merge conflicts, rebasing, and undo — practised on the
sandbox's seeded history.

| Module | What you learn | Time |
| ------ | -------------- | ---- |
| [Git & GitHub](git-github.md) | `main`/`staging`/`feature` model, PRs, conflicts, rebase, reflog | 75 min |

### Pillar 2 — Unit Testing
Writing tests you can trust: assertions, fixtures, parametrization, mocking, and
coverage.

| Module | What you learn | Time |
| ------ | -------------- | ---- |
| [Testing with pytest](02-testing-pytest.md) | assertions, fixtures, parametrization, mocking, `conftest.py` | 60 min |
| [Model & API tests](05-regression-model.md) → [FastAPI tests](06-fastapi-inference.md) | testing numeric code and a live API with `TestClient` | 45 min |

### Pillar 3 — CI/CD with GitHub Actions
Automate it all: run the suite on every push/PR across a version matrix, gate on
coverage, then build/test/publish the container image.

| Module | What you learn | Time |
| ------ | -------------- | ---- |
| [Continuous Integration](03-ci-github-actions.md) | GitHub Actions, build matrix, caching | 30 min |
| [CI/CD Pipeline](08-cicd-pipeline.md) | full suite + coverage in CI; build/test/publish image to GHCR | 40 min |

### Supporting modules (the substrate)
Context for the project the three pillars operate on — skim as needed.

| Module | What it covers |
| ------ | -------------- |
| [00 · Project Setup](00-project-setup.md) | venv, project layout, the Python-version/torch pitfall |
| [01 · Data Preprocessing](01-preprocessing.md) | the pandas/numpy transforms you will test |
| [04 · Docker Fundamentals](04-docker.md) | the image CI builds and publishes |
| [07 · Volumes & GPU](07-docker-volumes-gpu.md) | live-reload bind mounts, GPU passthrough |
| [09 · Troubleshooting](09-troubleshooting.md) | every real bug we hit and how we fixed it |

---

## Suggested agenda (one day)

1. **Setup** (00) — 20 min
2. **Pillar 1 — Git & GitHub** (git-github) — 75 min
3. **Pillar 2 — Unit Testing** (02, then 05→06 tests) — ~105 min
4. **Pillar 3 — CI/CD** (03, 08) — 70 min
5. **Wrap-up / troubleshooting** (09) — 20 min

Roughly a full day with breaks, or two half-days split after Pillar 1.

---

## The sandbox itself

This repo ships with a **seeded git history** — realistic branches and a planted
merge conflict — so the Git module has something real to work on. See the
[top-level README](../README.md) for the branching model and the list of
**planted exercises** (clean PR, conflict, rebase, red→green test, reflog
recovery).

Ready? Start with **[Module 00 — Project Setup](00-project-setup.md)**, then go
to **[Pillar 1 — Git & GitHub](git-github.md)**.

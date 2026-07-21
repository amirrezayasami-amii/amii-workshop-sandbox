# Module 03 — Continuous Integration with GitHub Actions

[← Testing](02-testing-pytest.md) · [Index](README.md) · [Next: Docker Fundamentals →](04-docker.md)

## Learning objectives

- Explain what CI is and why it matters.
- Author a GitHub Actions workflow that runs your test suite automatically.
- Use a **build matrix** to test across multiple Python versions.
- Speed up runs with dependency **caching**.
- Understand the git branching workflow used to keep changes reviewable.

## Prerequisites

- Modules 01–02 complete; `pytest -v` passes locally.
- A GitHub account (to see the workflow run for real).

---

## 1. What is CI?

**Continuous Integration** means every push and pull request automatically
builds the project and runs the tests, on a clean machine you don't control.
This catches:

- "Works on my machine" failures (missing dependency, wrong Python version).
- Tests that only pass because of local state.
- Regressions, *before* they merge.

GitHub Actions runs these jobs as **workflows** defined in YAML files under
`.github/workflows/`.

> **Where this is going.** In this module we build a **minimal** CI workflow
> that runs the preprocessing tests — the right starting point when the project
> is small. Once we've added the ML model, the API, and the Docker image
> (Modules 04–07), we **upgrade** it into a full CI/CD pipeline — coverage,
> CPU-only torch, and publishing the container image — in
> **[Module 08](08-cicd-pipeline.md)**. Learn the fundamentals here; see the
> production-grade version there.

---

## 2. The workflow file

Create `.github/workflows/tests.yml` (we'll rename and expand this into
`ci.yml` in Module 08):

```yaml
name: Tests

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13"]

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: pytest -v
```

### Line-by-line

| Section | Meaning |
| ------- | ------- |
| `name:` | Display name in the GitHub **Actions** tab. |
| `on:` | **Triggers.** Runs on pushes and PRs targeting `main`/`master`. Feature branches don't burn minutes on every push, but opening a PR does. |
| `runs-on: ubuntu-latest` | A fresh Ubuntu VM per job. |
| `strategy.matrix` | Runs the job **once per Python version**, in parallel. |
| `fail-fast: false` | If 3.11 fails, 3.12 and 3.13 still complete — so you see whether a failure is version-specific. |
| `uses: actions/checkout@v4` | Pulls a reusable action to clone your repo. `@v4` pins the major version. |
| `actions/setup-python@v5` | Installs the requested Python; `cache: pip` caches wheels between runs. |
| `${{ matrix.python-version }}` | GitHub expression syntax; substitutes the current matrix value. |
| `run:` | Shell commands. The `\|` is YAML for a multi-line script. |
| final `pytest -v` | **The job's pass/fail is this step's exit code.** A failing test → red check. |

> **Why 3.11–3.13 and not 3.14?** Same reason as Module 00: `numpy`/`pandas`
> wheels aren't reliable on 3.14 yet, so CI would fail building them. Pin to
> versions with stable wheels.

---

## 3. A `requirements.txt` as the single source of truth

CI installs dependencies from `requirements.txt`, so both local dev and CI use
the *same* list. At this stage it's just:

```
numpy
pandas
pytest
```

(We extend it in Module 05 with torch, fastapi, etc.)

---

## 4. The branching workflow

We kept each chunk of work on its own branch so changes stay small and
reviewable:

```bash
git checkout -b docker-setup      # start a feature branch off the current one
# ... make changes ...
git add -A
git commit -m "Add Docker setup"
```

Guidelines we followed:

- **Never commit straight to `main`** for feature work — branch first.
- **One logical change per commit**, with a descriptive message.
- Open a **pull request** to merge — that's what triggers CI on the PR.

> **Team convention note:** in this workshop's repo, commit messages omit any
> AI co-author trailer, per the maintainer's preference. Adopt whatever
> convention your team agrees on and apply it consistently.

---

## 5. Watch it run

1. Create a repo on GitHub and add it as a remote:

   ```bash
   git remote add origin git@github.com:<you>/<repo>.git
   git push -u origin master
   ```

2. Open the repo's **Actions** tab. You'll see the **Tests** workflow running,
   with three parallel jobs (3.11, 3.12, 3.13).

3. Open a pull request — the checks appear inline, and a red X blocks the merge
   until tests pass.

> **Reality check:** if there's no GitHub remote yet, the workflow file just
> sits in the repo doing nothing — it only executes on GitHub's servers. That's
> expected; it "activates" on your first push.

---

## Checkpoint ✅

- [ ] `.github/workflows/tests.yml` exists and is valid YAML.
- [ ] `requirements.txt` lists your dependencies.
- [ ] (If pushed) the Tests workflow runs green across 3.11–3.13.

## Exercises

1. Add a linting step (e.g. `pip install ruff` then `ruff check .`) as a new
   step before the tests.
2. Add a `workflow_dispatch:` trigger so you can run the workflow manually from
   the Actions tab.
3. Make a test fail on purpose, push to a branch, open a PR, and watch CI go
   red. Then fix it and watch it go green.

[← Testing](02-testing-pytest.md) · [Index](README.md) · [Next: Docker Fundamentals →](04-docker.md)

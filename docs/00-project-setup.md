# Module 00 — Project Setup

[← Back to index](README.md) · [Next: Data Preprocessing →](01-preprocessing.md)

## Learning objectives

By the end of this module you will be able to:

- Create an isolated Python virtual environment.
- Explain **why** isolation matters and what problem it solves.
- Recognize the two environment pitfalls we hit early (externally-managed
  Python and bleeding-edge interpreter versions).

## Prerequisites

- Python 3.11, 3.12, or 3.13 installed (see the pitfall about 3.14 below).
- Git installed and configured.

---

## 1. Why a virtual environment?

A virtual environment is a self-contained directory holding a specific Python
interpreter and its own set of installed packages. Without one, every
`pip install` mutates your **system-wide** Python, which leads to:

- Version conflicts between projects (project A needs `numpy 1.x`, project B
  needs `2.x`).
- "Works on my machine" bugs.
- On some systems, an outright refusal to install (see the next pitfall).

## 2. Create and activate the environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
```

Your prompt now shows `(.venv)`. Everything you `pip install` from here on lives
inside `.venv/` and nowhere else.

Upgrade pip and install the tools we start with:

```bash
python -m pip install --upgrade pip
pip install pytest numpy pandas
```

**Expected output (tail):**

```
Successfully installed numpy-2.x pandas-3.x pytest-9.x ...
```

Confirm:

```bash
python -c "import numpy, pandas, pytest; print('ok')"
```

---

## ⚠️ Pitfall 1 — "externally-managed-environment"

If you run `pip install` **without** a virtual environment on a Homebrew or
Debian/Ubuntu Python, you'll see:

```
error: externally-managed-environment
× This environment is externally managed
```

This is [PEP 668](https://peps.python.org/pep-0668/) protecting the
OS-managed Python from being corrupted by pip. **The correct fix is not
`--break-system-packages`** — it's to use a virtual environment, exactly as we
did above. This is *why* Module 00 exists before any code.

## ⚠️ Pitfall 2 — bleeding-edge Python versions

We initially had **Python 3.14** as the system interpreter. Several scientific
and ML packages (`torch` in particular) do not yet publish prebuilt "wheels"
for the newest interpreter, so `pip install torch` either fails or tries to
compile from source for a long time.

**Guidance for the workshop:** standardize attendees on **Python 3.12**. It has
stable wheels for everything we use (numpy, pandas, torch, fastapi). We keep the
CI matrix at 3.11–3.13 and the Docker image at 3.12 for exactly this reason.

> **Teaching tip:** this is a great real-world lesson — "newest" is not always
> "best" for a production stack. Dependency availability drives version choice.

---

## 3. Project directory layout

Create the working directory and initialize git (if not already a repo):

```bash
mkdir my-ml-workshop && cd my-ml-workshop
git init
```

We'll add files module by module. For now the important habit is: **one
concern per file**, and keep configuration (like `.gitignore`) in the repo root.

Create a starter `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/

# Virtual environments
.venv/
venv/
```

> **Why ignore `.venv/`?** It's large, machine-specific, and fully
> reproducible from `requirements.txt`. Committing it would bloat the repo and
> cause cross-platform breakage.

---

## Checkpoint ✅

You should now have:

- [ ] An activated `.venv` virtual environment.
- [ ] `pytest`, `numpy`, `pandas` installed inside it.
- [ ] A git repository with a `.gitignore`.

Verify in one line:

```bash
python -c "import numpy, pandas, pytest; print('environment ready')"
```

## Exercises

1. Deactivate (`deactivate`) and reactivate the environment. Confirm that
   `pip list` differs inside vs. outside the venv.
2. Intentionally run `pip install requests` **outside** the venv on a Homebrew
   Python and observe the PEP 668 error, then do it correctly inside the venv.

[← Back to index](README.md) · [Next: Data Preprocessing →](01-preprocessing.md)

# Pillar 1 — Git & GitHub

> **Goal:** be comfortable with the everyday team workflow — branch, commit,
> push, pull request, review, merge — and with the three things that scare
> people: **merge conflicts**, **rebasing**, and **undoing**.

This module is hands-on. Every exercise uses the *seeded history* already in
this sandbox, so you are practising on realistic branches instead of a blank
repo.

**Prerequisite:** you have cloned the repo and run `pytest -v` once (see
[Module 00](00-project-setup.md)). Set the graph alias so you can *see* history:

```bash
git config --global alias.graph "log --oneline --graph --decorate --all"
git graph
```

---

## Learning objectives

By the end you can:

- Explain the `main` / `staging` / `feature/*` branching model and why teams use it.
- Create a feature branch, commit with Conventional Commits, push, and open a PR.
- Resolve a merge conflict by hand and finish the merge.
- Rebase a stale feature branch and push it safely with `--force-with-lease`.
- Recover a "lost" commit with `git reflog`.

---

## 1. The branching model

```
main ────●───────────────●            stable, tagged (v0.1.0). Always green.
          \             /
staging ───●──●──●──●──●               integration branch. CI runs here.
               \   \
feature/*       ●   ●─●                your work. PR back into staging.
```

- **`main`** — production. You never commit here directly; releases are merged in.
- **`staging`** — where features integrate and stabilise. This is the default branch.
- **`feature/<name>`** — one branch per unit of work, branched off `staging`.

Look at what you were given:

```bash
git graph
git branch -a          # local + remote-tracking branches
```

---

## 2. Your first contribution (branch → commit → PR)

```bash
git switch staging
git switch -c feature/<your-name>        # branch off staging
```

Add your line to `PARTICIPANTS.md`, then:

```bash
git add PARTICIPANTS.md
git commit -m "feat: add <your-name> to participants"
git push -u origin feature/<your-name>
```

On GitHub, click **Compare & pull request**, set the base to **`staging`**, and
open it. Congratulations — that is 90% of the daily workflow.

> **Conventional Commits.** `<type>: <summary>` where type is one of
> `feat, fix, docs, refactor, style, test, chore, ci`. It keeps history
> readable and lets tools generate changelogs.

---

## 3. Exercise — open a clean PR

**Branch:** `feature/standardize-columns`

This branch adds a ready-to-ship change (it wires `standardize()` into the
preprocessing pipeline). Review it and merge it:

```bash
git switch feature/standardize-columns
git graph                                # see how it sits ahead of staging
git log staging..HEAD --oneline          # exactly what this PR would add
```

Push it (if it is local-only) and open a PR into `staging`, or merge locally to
see a fast-forward/merge commit:

```bash
git switch staging
git merge --no-ff feature/standardize-columns
```

**Checkpoint:** `git graph` shows the feature branch merged into `staging` and
`pytest -v` is still green.

---

## 4. Exercise — resolve a merge conflict

**Branch:** `feature/tune-lr` · **Conflict file:** `config.py`

Two people changed the **same line**. `staging` lowered `LEARNING_RATE` for
stability; `feature/tune-lr` raised it to train faster. Git cannot pick for you.

```bash
git switch staging
git merge feature/tune-lr
# CONFLICT (content): Merge conflict in config.py
```

Open `config.py`. You will see:

```python
<<<<<<< HEAD
LEARNING_RATE = 5e-4          # staging: lowered for stability
=======
LEARNING_RATE = 3e-3          # feature/tune-lr: raised to train faster
>>>>>>> feature/tune-lr
```

Decide on the real value (delete the markers, keep one line — or write a
compromise), then finish the merge:

```bash
git add config.py
git commit                    # completes the merge with a merge commit
pytest -v                     # confirm nothing broke
```

> **Tip:** `git merge --abort` at any time returns you to before the merge.
> Prefer resolving in your editor's 3-way merge view or `git mergetool`.

**Checkpoint:** no `<<<<<<<` markers remain (`git grep -n "<<<<<<<"` is empty)
and the suite passes.

---

## 5. Exercise — rebase safely

**Branch:** `feature/faster-epochs`

This branch was created *before* `staging` moved ahead, so its history is stale.
Rebasing replays your commits on top of the current `staging` for a linear
history:

```bash
git switch feature/faster-epochs
git rebase staging
#   ...resolve any conflict, then: git rebase --continue...
```

Because rebase **rewrites commits**, pushing an already-pushed branch needs a
force — but use the *safe* one, which refuses if someone else pushed meanwhile:

```bash
git push --force-with-lease
```

> **Golden rule:** rebase your *own* feature branches freely; never rebase
> shared branches (`main`, `staging`) that others build on.

---

## 6. Exercise — undo & recover

Nothing in git is truly lost for ~30 days. Practise the scary commands here:

```bash
git switch -c scratch/undo-demo
echo "throwaway" >> PARTICIPANTS.md && git commit -am "chore: throwaway commit"
git reset --hard HEAD~1          # the commit is "gone"
git reflog                       # ...but here it is
git reset --hard <hash-from-reflog>   # bring it back
```

Other everyday undo tools:

| You want to… | Command |
| ------------ | ------- |
| Unstage a file | `git restore --staged <file>` |
| Discard working-tree changes | `git restore <file>` |
| Amend the last commit message | `git commit --amend` |
| Undo a *pushed* commit without rewriting history | `git revert <hash>` |

---

## Checkpoint (end of pillar 1)

You have: opened a PR into `staging`, resolved a conflict in `config.py`,
rebased a stale branch and force-pushed it safely, and recovered a commit with
reflog. Next: **[Pillar 2 — Unit Testing](02-testing-pytest.md)**, where the
tests you rely on to merge with confidence are written.

## Exercises to extend

1. Protect `main` on GitHub (Settings → Branches) so PRs are required.
2. Add a `CODEOWNERS` file so your PRs auto-request a reviewer.
3. Squash-merge a multi-commit feature branch and compare the resulting history.

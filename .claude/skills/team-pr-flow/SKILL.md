---
name: team-pr-flow
description: The trash-sorter team's rules for getting code into main. Use when starting work on an issue, or when committing, pushing, or opening a pull request in this repo.
---

# Team PR flow

`main` is protected: every change reaches it through a pull request that passes CI (`test` job) and gets one approval from a teammate. Run the stage that matches where the user is; each stage ends on its completion criterion.

## Stage 1: start work on an issue

1. Find the issue number. If the work has no issue yet, ask the user whether to open one (`gh issue create`, template "工作項目 / Task", label the matching `area:*`). Every branch traces back to an issue.
2. Branch from a fresh `main`:

   ```sh
   git fetch origin
   git switch -c <issue>-<short-kebab-slug> origin/main   # e.g. 12-convert-glass-assets
   ```

   If the working tree has uncommitted changes, ask the user before carrying them over (`git stash` → switch → `git stash pop`).
3. Claim the issue and move its card to In Progress:

   ```sh
   gh issue edit <issue> --add-assignee @me
   ```

   Then the board (skip with a note to the user if `gh` lacks the `project` scope):

   ```sh
   item=$(gh project item-list 1 --owner nuume-trash-classification --format json --limit 500 \
     --jq '.items[] | select(.content.number==<issue>) | .id')
   gh project item-edit --project-id PVT_kwDOFJ4tn84BpJ7W --id "$item" \
     --field-id PVTSSF_lADOFJ4tn84BpJ7WzhnmzPg --single-select-option-id 43525905   # In Progress
   ```

Done when: you are on `<issue>-<slug>` based on current `origin/main`, and the issue is assigned.

## Stage 2: commit

1. Run the checks and fix what they report:

   ```sh
   uv run ruff check --fix .
   uv run ruff format .
   uv run pytest
   ```

2. Stage files by name (`git add <path>…`). Commit only source, tests, docs, and small config. Datasets, weights (`*.pt`, `*.onnx`), renders, generated USD, and `.env` stay out of git; `.gitignore` covers most of them, so review `git status` for anything it misses and share large files by drive link in the PR instead.
3. Commit message in English, imperative, one focused change per commit (`Add glass material to asset converter`).

Done when: ruff and pytest are clean and `git status` shows only the intended changes committed.

## Stage 3: push and open the PR

1. Rebase onto the latest main if it has moved, rerun pytest, then push:

   ```sh
   git fetch origin && git rebase origin/main
   git push -u origin HEAD
   ```

2. Open the PR. The title is English; the body follows `.github/pull_request_template.md` (Chinese is fine) and contains `Closes #<issue>`. Fill "怎麼驗證" with the commands you ran and their results; simulation or training changes carry numbers (pick success rate, miss rate, mask mAP50).

   ```sh
   gh pr create --title "<English title>" --body-file <filled template> --reviewer <teammate>
   ```

   Ask the user who should review if they have not said.
3. Move the issue card to In Review (same command as stage 1, option id `daf0808b`).
4. Watch CI: `gh pr checks --watch`. If `test` goes red, fix on the same branch and push again.

Done when: the PR is open, linked to its issue, has a reviewer, and `test` is green.

## Review and merge

- Review feedback: commit fixes on the same branch and push; the PR updates itself.
- Merging is the reviewer's or author's call after approval, with **Squash and merge**. Merge only when the user asks and the PR shows an approval from a teammate; the branch protection enforces the same.

## Project guardrails

These come from `docs/design.md` and hold for every change:

- **Privileged-information rule**: the `--perception yolo` path uses camera images (plus belt speed) only. Ground-truth object poses or classes from the simulator appear only in `--perception gt` mode and in scoring.
- **Test-set isolation**: `data/real/test` is used once, for the final evaluation. Training, tuning, and threshold choices use synthetic data and `data/real/finetune`.

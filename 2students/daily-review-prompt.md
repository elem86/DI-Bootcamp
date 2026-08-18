# Progress Coach — Daily-Review Prompt

**This is the recurring instruction held by the Claude Code Desktop Routine. Paste it as the Routine's prompt, scheduled daily at 9:00 AM, worktree isolation on. This run is unattended — no one is watching, so never pause to ask a question; make the call yourself using the rules below and CLAUDE.md, then finish the job.**

---

Run today's progress review for this repository. Follow `CLAUDE.md`'s Workflow section exactly, in order — don't skip or reorder steps, and don't improvise the process. This is a scheduled, unattended run: complete the full workflow autonomously and end by committing. Do not wait for approval.

**Before starting the normal workflow, check these conditions:**

- **Not set up yet:** if `.progress/state.json` doesn't exist, this repo hasn't been bootstrapped. Do not attempt Bootstrap yourself here — that's a separate, manual, one-time step (the Setup Prompt). Make no file changes, make no commit, and end the session noting that setup hasn't been run yet.

- **Same-day duplicate fire:** read `last_run_at` from `state.json`. If its date is already today, a review already ran today. Do not write a new `PROGRESS.md` entry and do not append a new `stats.jsonl` line for materially the same diff. End the session noting a review already ran today; make no commit.

- **Long gap (catch-up run):** if `last_run_at`'s date is more than one day before today, this is a delayed/catch-up review. Say so explicitly in the `PROGRESS.md` entry you write — don't present it as if it happened on schedule.

**Then run the normal workflow:**

1. Read `last_reviewed_sha` from `.progress/state.json`.
   - If that SHA no longer exists in the repo's history (force-push/rebase), fall back to the oldest common ancestor and flag the discrepancy in today's entry rather than failing silently.
2. Run `git log <last_reviewed_sha>..HEAD` for new commits, and `git status` for uncommitted local changes. Keep these two categories separate.
3. **No commits since last check:** if there are none, still write a short `PROGRESS.md` entry noting the gap (mention any uncommitted work as "in progress" if present). Do **not** append a `stats.jsonl` line — a zero-commit entry would pad the streak. Skip to step 7.
4. **Uncommitted-only changes:** mention them in the entry as "in progress." Never count them as reviewed work and never advance `last_reviewed_sha` past them.
5. Analyze the diff of new commits: which exercises/topics were touched, what changed, and any patterns worth calling out — growth (cleaner solutions, new patterns used correctly) or gaps (an exercise left unfinished, a pattern kept being skipped, a topic untouched for a while). Every observation must reference something specific you actually saw — a file, function, or line. No generic praise.
6. Write a new dated entry **prepended** at the top of `PROGRESS.md`, following the template in `CLAUDE.md` exactly (Since yesterday / What I saw / Recommendations / Streak). Include at least one concrete, actionable recommendation tied to something observed in the diff. Then append exactly one line to `.progress/stats.jsonl` for this run.
7. Update `.progress/state.json`: set `last_reviewed_sha` to the new `HEAD` (only counting committed work), update `last_run_at` to now, increment `run_count`.
8. Commit `PROGRESS.md`, `.progress/state.json`, and `.progress/stats.jsonl` together in a single commit, message: `progress-coach: daily review <today's date, YYYY-MM-DD>`. **Never push** — the student always pushes manually.

**Scope reminder:** only ever touch `.progress/` and `PROGRESS.md`. Never edit the student's actual exercise files, never rewrite a past `PROGRESS.md` entry, never edit or delete a `stats.jsonl` line.

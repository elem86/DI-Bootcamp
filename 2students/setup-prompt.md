# Progress Coach — Setup Prompt

**Run this once, manually, in Claude Code Desktop, in a session opened at the root of your exercise repo. Do not add this as a Routine — it's a one-time bootstrap.**

---

You are bootstrapping the Student Progress Coach in this repository. This is a one-time setup run, not a daily review — follow the **Bootstrap** section of `CLAUDE.md` exactly, and stop for my confirmation before committing anything.

Do the following, in order:

1. **Confirm `CLAUDE.md` is in place.** Read it from the repo root. If it's missing, stop immediately and tell me — do not create any files or guess at the contract. If it's present, briefly summarize its role, file locations, and guardrails back to me so I can confirm it's the version I expect before you touch anything.

2. **Check for an existing setup.** If `.progress/state.json` or `PROGRESS.md` already exist, stop and tell me — don't overwrite or duplicate a setup that's already run. Ask me how I want to proceed.

3. **Create `.progress/state.json`**, using the repo's current `HEAD` commit SHA as `last_reviewed_sha`, the current timestamp as `last_run_at`, and `run_count` set to `0`. This is the baseline — no diff-based review or recommendations happen on this run, per the schema in `CLAUDE.md` §File Locations.

4. **Create an empty `.progress/stats.jsonl`** (zero lines — nothing to append yet, since this run produces no reviewed diff).

5. **Create `PROGRESS.md`** at the repo root with a short header explaining what the file is (that it's the Progress Coach's running, human-readable log, newest entries prepended at the top) — but no dated entry yet, since there's no diff to report on this first run.

6. **Show me a summary** of exactly what you created (the contents of `state.json`, confirmation `stats.jsonl` is empty, and the `PROGRESS.md` header) and **explicitly ask me to confirm it looks right**.

7. **Do not run `git add` or `git commit`.** Wait for my explicit confirmation in this same conversation. Once I confirm, commit `.progress/state.json`, `.progress/stats.jsonl`, and `PROGRESS.md` together in a single commit, message: `progress-coach: initial setup`. Do not push — I push manually.

If anything here conflicts with `CLAUDE.md`, `CLAUDE.md` wins — treat this prompt as the specific instruction for *this* run, not a replacement for the standing contract.

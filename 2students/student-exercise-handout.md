# Student Progress Coach — Exercise Handout
**Day 3 Capstone — Claude Code Desktop Routines**

You're building yourself an automated coach that checks your course repo every morning, tells you what you actually did, gives you specific feedback, and tracks your progress as a running history — without your instructor manually reading every repo.

You're not writing this from a blank page. You're given a working starting point (`CLAUDE.md`, `setup-prompt.md`, `daily-review-prompt.md`) built from the project spec (`student-progress-coach-spec.md`). Your job today is to understand *why* it's built the way it is, adapt it to your own repo, and get it running end to end. That process — reading a behavior contract critically, testing prompts before trusting them with a schedule, and verifying output against a spec — is the actual skill this exercise teaches.

## Before you start

- Have your course exercise repo cloned locally, with at least a few real commits in its history.
- Have Claude Code Desktop installed and able to open a session at your repo's root.
- Read `student-progress-coach-spec.md` once, start to finish, before touching any files. You don't need to memorize it — you need to know it exists and what section covers what, because you'll refer back to it.

## Step 1 — Understand what you were handed

Open `CLAUDE.md`. This is the behavior contract the coach follows every time it runs, including today's setup run. Before changing anything, check it against the spec's requirements in **§7 CLAUDE.md Requirements**:

- Does it define a Role? A Repo Context? File locations and schemas matching §6? The exact workflow from §4? The output template from §6.3? Guardrails matching §11?

If you can't find where CLAUDE.md covers one of those, that's worth flagging — either it's there and you missed it, or it's a real gap. Don't assume the starting materials are perfect; reading them critically is the point.

## Step 2 — Make it yours

`CLAUDE.md` is written generically. Skim it for anything that should reflect your actual repo or preferences — for example, the "Repo Context" language, or example topics in the schema samples. You don't need to rewrite structure, just make sure nothing in it describes a repo that isn't yours.

Place (or confirm) `CLAUDE.md` at your repo's root.

## Step 3 — Run the setup prompt (once, manually)

Open a Claude Code Desktop session at your repo root. Paste in the full contents of `setup-prompt.md` as your first message.

This is a one-time bootstrap — do **not** turn this into a Routine. Watch for:

- Claude reading `CLAUDE.md` back to you before doing anything. If it doesn't, stop and ask why.
- Claude stopping *before* committing, showing you what it created, and asking you to confirm.

**Checkpoint:** after this step, `.progress/state.json`, `.progress/stats.jsonl` (empty), and `PROGRESS.md` (header only, no dated entry) should exist. Confirm they look right before approving the commit.

## Step 4 — Test the daily-review prompt manually

Don't schedule anything yet — a scheduled prompt you haven't tested is a prompt you don't understand. Make a couple of small real commits to your repo (something to actually review), then start a **new** Claude Code Desktop session at your repo root and paste in the full contents of `daily-review-prompt.md`.

Check the result against the spec:

- Does the new `PROGRESS.md` entry follow the template exactly (§6.3)?
- Is the feedback specific — does it name an actual file or pattern from your diff, not generic praise? (§7 Role, FR-2/FR-3)
- Did it append exactly one line to `stats.jsonl`, and update `state.json` correctly?
- Did it commit with a `progress-coach:`-prefixed message, and *not* push?

If something's off, this is the moment to fix it — either in `daily-review-prompt.md` or `CLAUDE.md` — before it's running unattended on a schedule.

## Step 5 — Create the Routine

In Claude Code Desktop, create a new Routine with:

- **Prompt:** the full contents of `daily-review-prompt.md`
- **Schedule:** daily, 9:00 AM
- **Worktree isolation:** ON — this keeps the scheduled run from colliding with whatever you're actively editing
- **Permission mode:** Manual for the first run, so you can see and approve what it does before trusting it to auto-run

## Step 6 — Observe one full run

Use "Run now" to trigger the Routine on demand rather than waiting until tomorrow morning. Confirm it produces a correctly formatted `PROGRESS.md` entry and exactly one new `stats.jsonl` line, and that it committed (without pushing).

## Definition of Done

Check these off as you complete them — this is exactly the spec's §12 acceptance criteria:

- [ ] `CLAUDE.md` reviewed against spec §7 and adapted to my repo
- [ ] Setup prompt run once; `.progress/` and `PROGRESS.md` exist and look correct
- [ ] Daily-review prompt tested manually against a real diff, output checked against the template
- [ ] Routine created: daily 9:00 AM, worktree isolation on
- [ ] One full scheduled run observed via "Run now," producing a correct `PROGRESS.md` entry and one `stats.jsonl` line

## Reflection (optional, if time allows)

- Which edge case in the spec's §10 table do you think you're most likely to actually hit in the next few weeks?
- If you had to change one guardrail in `CLAUDE.md`, what would it be and why?
- The spec explicitly puts "on-request trend summary across `stats.jsonl`" out of scope for today's build (§13-adjacent). If you finish early, try writing a short prompt that reads `stats.jsonl` and summarizes your streak, topic coverage, and velocity — no CLAUDE.md changes needed, this can be a standalone ad-hoc prompt.

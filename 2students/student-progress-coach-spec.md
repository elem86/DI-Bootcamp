# Student Progress Coach — Feature Spec
**Day 3 Capstone Project — Claude Cowork + Claude Code**

Status: Draft for course build
Owner: Instructor (Max)
Builders: Students, during Day 3

---

## 1. Purpose

Each student has a local (and GitHub-hosted) repository of course exercises. This project gives every student their own automated coach that checks their repo each morning, summarizes what they did since the last check, gives concrete recommendations, and tracks their progress as a running history — without the instructor manually reviewing every repo.

Secondary purpose: the *build itself* is the Day 3 exercise. Students aren't just using a prebuilt tool — they write the CLAUDE.md and the two prompts that make it work, so the deliverable teaches CLAUDE.md design, prompt design for unattended/scheduled agents, and Claude Code Desktop Routines.

## 2. Scope

**In scope (v1)**
- Detect what changed in a student's repo since the last check (new commits, changed files)
- Produce a written summary + specific recommendations
- Persist stats across days so progress is visible as a trend, not a single snapshot
- Run automatically every morning without the student manually invoking it

**Out of scope (v1)** — see §13 for future paths
- Cross-student comparison or instructor dashboard
- Delivery to Slack/email (Cowork could do this later)
- Triggering off GitHub push events instead of a fixed time
- Scoring against a fixed rubric or grading

## 3. Users

| User | Role |
|---|---|
| Student | Owns the repo and the Routine; reads `PROGRESS.md` each morning |
| Instructor | Defines the CLAUDE.md/prompt template used as the class starting point; does not need direct access to run it |

## 4. System Overview

**Components**

1. **Exercise repo** — local clone, pushed to GitHub periodically. Source of truth for "what was done."
2. **`.progress/state.json`** — tracks the last commit SHA the coach reviewed, so each run only looks at *new* work.
3. **`.progress/stats.jsonl`** — append-only log, one line per run, the raw data behind "monitor progress over the days."
4. **`PROGRESS.md`** — human-readable output at repo root; a single file that grows for the whole course (see §13 for upgrade paths once it gets long); what the student actually reads over coffee.
5. **`CLAUDE.md`** — the behavior contract: role, tone, data locations, output format. Loaded automatically every run.
6. **Setup prompt** — run once, manually, to bootstrap the above files.
7. **Daily-review prompt** — the recurring instruction, held by the Routine, executed every morning.
8. **Execution engine: Claude Code Desktop local Routine** — has direct filesystem/git access (required, since state includes local uncommitted work, not just what's on GitHub); runs on a daily schedule; can commit results back to the repo.

**Data flow (one run)**

```
Routine fires (scheduled time)
  → Claude Code Desktop starts a session in the repo, loads CLAUDE.md
  → reads .progress/state.json → gets last_reviewed_sha
  → git log <last_reviewed_sha>..HEAD  (+ git status for uncommitted work)
  → analyzes diff: what topics/exercises, code quality notes, gaps
  → writes new dated section to PROGRESS.md
  → appends one line to .progress/stats.jsonl
  → updates .progress/state.json with new HEAD sha + timestamp
  → commits these files with a distinct marker (see §11)
```

## 5. Functional Requirements

| ID | Requirement | Acceptance criteria |
|---|---|---|
| FR-1 | Detect new work since last check | Correctly identifies all commits after `last_reviewed_sha`; also flags uncommitted local changes separately, labeled as "in progress" |
| FR-2 | Summarize the work | Summary names specific files/exercises touched, not generic phrasing |
| FR-3 | Give recommendations | At least one concrete, actionable suggestion per run, tied to something actually observed in the diff — never a generic "keep it up" |
| FR-4 | Persist stats | Every run appends exactly one entry to `stats.jsonl`; no entry is ever edited or deleted by the coach |
| FR-5 | Show progress over time | On request (or periodically), the coach can summarize trends across multiple stats entries (streak, topic coverage, velocity) |
| FR-6 | Run unattended on a schedule | Executes daily without the student starting it manually |

## 6. Data Model

### 6.1 `.progress/state.json`
```json
{
  "last_reviewed_sha": "a1b2c3d",
  "last_run_at": "2026-08-18T07:02:00+03:00",
  "run_count": 12
}
```

### 6.2 `.progress/stats.jsonl` (append-only, one JSON object per line)
```json
{"date":"2026-08-18","commits":4,"files_changed":6,"lines_added":142,"lines_removed":31,"topics":["dictionaries","error handling"],"streak_days":5,"recommendation_count":2}
```

### 6.3 `PROGRESS.md` entry template

Since this is one file that grows for the whole course, new entries are prepended at the top (newest day first) so the student sees today's entry without scrolling past prior history.

```markdown
## 2026-08-18

**Since yesterday:** 4 commits, 6 files — dictionaries, error handling

**What I saw:** [2-4 sentences, specific to the actual diff]

**Recommendations:**
- [specific, tied to an observed file/line/pattern]
- [specific]

**Streak:** 5 days in a row
```

## 7. CLAUDE.md Requirements

The CLAUDE.md must define, at minimum:
- **Role** — what kind of coach this is (encouraging but substantive; feedback must reference actual code, not generic praise)
- **Repo context** — this is a learning exercise repo, not production code; review accordingly
- **File locations** — exact paths for state, stats, and output files, and their schemas (§6)
- **Workflow** — the exact sequence in §4, so the model doesn't improvise the process each run
- **Output format** — the `PROGRESS.md` template, followed exactly so entries stay parseable/consistent
- **Guardrails** — never rewrite past `PROGRESS.md` entries or edit `stats.jsonl` history; only append; never force-push

## 8. Prompt Requirements

### 8.1 Setup prompt (run once, manually)
Must direct Claude to:
- Create `.progress/` with an initial `state.json` (using current HEAD as baseline — no diff generated on this first run) and empty `stats.jsonl`
- Create `PROGRESS.md` with a short header explaining what it is
- Confirm the CLAUDE.md is in place and ask the student to review it before proceeding
- Not commit anything until the student confirms the setup looks right

### 8.2 Daily-review prompt (the Routine's instructions)
Must direct Claude to:
- Follow the CLAUDE.md workflow exactly
- Handle the "nothing changed" case explicitly (see §10)
- Handle the "long gap" case explicitly — if the last run was more than a day ago (e.g., a missed/catch-up run), say so rather than presenting a stale summary as if it were today's
- End by committing `PROGRESS.md`, `state.json`, and `stats.jsonl` together with a clearly labeled commit message (e.g. `progress-coach: daily review 2026-08-18`)
- Never push to GitHub — the student always pushes manually. See §11.

## 9. Scheduling & Runtime

- **Engine:** Claude Code Desktop, local Routine (not Cowork, not cloud Routine — see the Day 3 outline discussion for why: local access to uncommitted work is required)
- **Cadence:** Daily at 9:00 AM, uniform across the whole class
- **Worktree isolation:** on — the run should never collide with a student's in-progress uncommitted edits in their main working copy
- **Permission mode:** first run in Manual mode so the student can approve tool use once; subsequent runs auto-approve the same tools
- **Bootstrap:** first-ever run has no `last_reviewed_sha` — treat entire repo history (or just current HEAD) as baseline, not as "new work to review"
- **Idempotency:** if the Routine fires twice for the same day (e.g., a catch-up run after a missed one), it must not create two stats entries for materially the same diff — check `last_run_at` date before appending

## 10. Edge Cases

| Case | Expected behavior |
|---|---|
| No commits since last check | Still write a short `PROGRESS.md` entry noting the gap; no stats entry with zero commits padding the streak |
| Uncommitted local changes only | Mention as "in progress," don't count as reviewed work, don't advance `last_reviewed_sha` |
| Missed run (laptop asleep) catches up late | Note explicitly that this is a delayed/catch-up review, don't imply it happened at the scheduled time |
| Student rewrites history (force-push, rebase) | If `last_reviewed_sha` no longer exists in history, fall back to comparing against the oldest common ancestor and flag the discrepancy rather than failing silently |
| First run ever | Bootstrap only — no diff-based review, no recommendations yet |

## 11. Non-Functional Requirements

- **Tone/honesty:** feedback must be specific and evidence-based; no generic encouragement not tied to an observed change
- **Data hygiene:** `stats.jsonl` is append-only, never edited or reordered
- **Git hygiene:** coach commits are clearly distinguishable from the student's own exercise commits (dedicated commit message prefix); coach never pushes to the remote — the student always pushes manually
- **Scope discipline:** coach only touches `.progress/` and `PROGRESS.md` — never edits the student's actual exercise files

## 12. Acceptance Criteria / Definition of Done (for the Day 3 build)

- [ ] CLAUDE.md written and covers all items in §7
- [ ] Setup prompt tested manually once; `.progress/` and `PROGRESS.md` exist and look correct
- [ ] Daily-review prompt tested manually at least once against a real diff before scheduling
- [ ] Routine created in Claude Code Desktop, correct cadence, worktree isolation on
- [ ] One full scheduled run observed (via "Run now") producing a correctly formatted `PROGRESS.md` entry and one `stats.jsonl` line

## 13. Future Enhancements (explicitly out of scope for v1)

- Cloud Routine triggered on GitHub push events, so review happens per-push instead of once a day
- Cowork scheduled task to deliver the daily summary via email/Slack in addition to the repo file
- Weekly/aggregate trend report generated from `stats.jsonl` across the whole course
- Optional instructor-facing rollup across all students' `stats.jsonl` files (would need explicit student opt-in)
- **`PROGRESS.md` upgrade paths**, once the single growing file gets unwieldy:
  - Archive entries older than a few weeks into `progress-archive/`, keeping only recent history in the active file
  - Split into per-month files (e.g. `progress-2026-09.md`) with a short index at the root
  - Auto-generate a "this week" rollup at the very top instead of relying on prepended daily entries alone

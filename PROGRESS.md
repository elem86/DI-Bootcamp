# Progress Log

This is the running log kept by the Student Progress Coach (see `2students/CLAUDE.md`). Each daily review adds one dated entry, prepended to the top of this file — newest first — so today's review is always the first thing you see. Entries are never edited or reordered after the fact.

---

## 2026-08-25

**Since last review:** 0 commits, 0 files — no exercise topics

**What I saw:** Still no new commits under a `WeekN/DayM` folder — `git log a2a0d54..HEAD` shows only the coach's own Aug 24 daily-review commit (`0178e0a`), which touched just `PROGRESS.md` and `.progress/state.json`. The last real exercise commit stays `5fc2295` from Aug 19, so that's six days quiet now (Aug 20–25). **Correction to the last three reviews:** `DI_198/` and `hack1_movie_rec/` are not "uncommitted work sitting on disk" — both are their own separate git repositories (each has its own `.git/`), so the outer repo's `git status` only ever showed the folder name, never their contents. `hack1_movie_rec/.git/config` points at `origin = https://github.com/elem86/hack1_movie_rec.git`, and its `main` branch has a real history: "test", "downloaded movie database", "finished EDA", a merge of a `recommendation-engine` branch, a merge of a `final-demo` branch, "removed absolute paths", and "added gitignore" as the last commit — all dated Aug 13–16, before this coach's tracking even started on Aug 18. It's a finished, pushed hackathon project, not neglected work. `DI_198/` is different: it's a single-commit clone of `https://github.com/stolovitskyinc-maker/DI_198.git` from around July 10, with no local commits since — reference/backup material, not in-progress work. `current_position` stays at `Week6/Day5`.

**Recommendations:**
- Six days without a commit under the tracked `WeekN/DayM` structure — even a small one at `Week6/Day5` or wherever the course has actually moved to would restart the streak.
- Now that `hack1_movie_rec` is confirmed finished and pushed, it's worth deciding whether it belongs inside this tracked repo at all (as a submodule or a note) so future reviews don't re-flag it as missing — right now it'll keep showing up as an untracked folder forever since it's a separate repo.

**Streak:** 0 days (last exercise commit was Aug 19; nothing since)

---

## 2026-08-24

**Since last review:** 0 commits, 0 files — no exercise topics

**What I saw:** Another quiet day on the exercise side. `git log 86378da..HEAD` shows only the coach's own Aug 23 daily-review commit (`a2a0d54`) between the last review and now — it touched just `PROGRESS.md` and `.progress/state.json`, nothing under a `WeekN/DayM` folder. The last real exercise commit is still `5fc2295` from Aug 19, so that's now five days quiet (Aug 20–24). `DI_198/` and `hack1_movie_rec/` remain exactly as they were the last two reviews: both still fully uncommitted at the repo root, no new files or changes inside either. `current_position` stays at `Week6/Day5`.

**Recommendations:**
- Five days without an exercise commit — even a small one under the current `WeekN/DayM` position (or wherever the course has moved to) would restart the streak and give the next review something real to look at.
- `hack1_movie_rec/`'s three notebooks and `recommender.py` are still invisible to git — a WIP commit would at least put a checkpoint in history instead of leaving it all only on disk.

**Streak:** 0 days (last exercise commit was Aug 19; nothing since)

---

## 2026-08-23

**Since last review:** 0 commits, 0 files — no exercise topics

**What I saw:** Still nothing new on the exercise side. The only commit between the last two reviews (`0dc3861` → `HEAD`) is the coach's own Aug 22 daily-review commit (`86378da`), which only touched `PROGRESS.md` and `.progress/state.json`. The last actual exercise commit remains `5fc2295` from Aug 19 — that's now four days quiet (Aug 20–23). `DI_198/` (a full `week_1`…`week_8` plus `Hackathon/LaunchPad_AI` tree) and `hack1_movie_rec/` (three notebooks — `01_data_exploration.ipynb`, `02_recommendation_system.ipynb`, `03_final_demo.ipynb` — plus a `recommender.py`) are both still sitting fully uncommitted at the repo root, unchanged in scope from the last two reviews. `current_position` stays at `Week6/Day5`.

**Recommendations:**
- Four days without an exercise commit now — even a small commit under wherever the course has actually moved to would restart the streak and give this review something concrete to look at.
- If `hack1_movie_rec/` is at a stopping point, committing it (even as a WIP commit) would stop it being invisible to this tracker — right now the three notebooks and `recommender.py` exist on disk but nothing about them is in git history.

**Streak:** 0 days (last exercise commit was Aug 19; nothing since)

---

## 2026-08-22

*(Delayed/catch-up review — the last daily run was 2026-08-20; this covers everything committed since then.)*

**Since last review:** 0 commits, 0 files — no exercise topics

**What I saw:** No new exercise commits landed between `5fc2295` (Aug 19) and now. The only git activity in that window is the coach's own bookkeeping: the Aug 20 daily review commit (`18fa277`) and, earlier today, the weekly summary commit (`0dc3861`) — neither touches exercise code. The two untracked project folders at the repo root, `DI_198/` and `hack1_movie_rec/`, are still sitting there uncommitted; `DI_198/` is its own nested git repo (with its own `week_1`…`week_8` structure) and `hack1_movie_rec/` looks like a separate hackathon project. Both remain outside this course repo's `WeekN/DayM` structure, so they're noted but not reviewed as exercise work. `current_position` stays at `Week6/Day5`.

**Recommendations:**
- Nothing new to review, so the standing recommendation holds: get back to committing exercise work under a `WeekN/DayM` folder — it's been three days (Aug 20–22) since the last one.

**Streak:** 0 days (last exercise commit was Aug 19; nothing since)

---

## Week of 2026-08-16 — Weekly Summary

**This week:** 5 commits across 2 of 7 days, 8 files — dictionaries, error handling

**Highlights:** The week's one substantive exercise commit was `ebda6d9` ("testing the daily review"), which added `Week1/Day1/exercises_XP.py`: a dict comprehension in `word_lengths()` and a `safe_divide()` wrapped in try/except that returns `None` on `ZeroDivisionError` instead of crashing — clean, idiomatic use of both patterns. Everything else committed this week (Aug 18–19) was repo/coach scaffolding: reorganizing `2students/CLAUDE.md`, moving `.progress/` into place, and creating `Wek8/exercise.ipynb`, which is still an empty 0-byte placeholder that doesn't match the `WeekN/DayM` naming pattern needed to register as progress. No exercise work has been committed since Aug 19 — Aug 20 through 22 are quiet.

**Recommendations:**
- Rename `Wek8/` to a proper `WeekN/DayM` path (e.g. `Week8/Day1/`) and put real content in `exercise.ipynb` — it's been sitting empty all week.
- Get back to committing exercise work — the streak broke after Aug 19 and it's now been three days (Aug 20–22) without a new exercise commit.

**Streak:** 0 days (last exercise commit was Aug 19; nothing since)

---

## 2026-08-20

*(Delayed/catch-up review — the last run was 2026-08-18; this covers everything committed since then.)*

**Since last review:** 1 commit, 4 files — no exercise topics (repo/spec maintenance)

**What I saw:** The one commit since last time (`5fc2295`, "dailysummary") didn't touch exercise code — it edited `2students/CLAUDE.md` itself (adding the `current_position` tracking rule and the "don't guess a misnamed folder into current_position" guardrail), added a `current_position` field to `.progress/state.json` by hand, and added a `.gitignore` for `.coach-scripts/`. The same commit also created `Wek8/exercise.ipynb`, which is empty (0 bytes) and, fittingly, is exactly the kind of misnamed folder the new CLAUDE.md rule just warned about: it's missing the `Week` prefix (should be `Week8` or similar) and has no `DayM` subfolder, so it doesn't count as a real `WeekN/DayM` exercise. Per that rule, `current_position` stays at `Week6/Day5`, the last folder that matched the pattern. There are also two untracked project directories at the repo root (`DI_198/`, `hack1_movie_rec/`) that look like separate, unrelated projects rather than course exercises, so they're outside this review's scope and haven't been touched.

**Recommendations:**
- Rename `Wek8/` to a proper `WeekN/DayM` path (e.g. `Week8/Day1/`) and add real content to `exercise.ipynb` — right now it's an empty placeholder that won't register as progress.

**Streak:** 2 days in a row (Aug 18–19)

---

## 2026-08-18 (follow-up)

**Since earlier today:** 1 exercise commit, 1 file — dictionaries, error handling

**What I saw:** Real exercise work arrived. Your `Week1/Day1/exercises_XP.py` shows two solid fundamentals: `word_lengths()` uses a dict comprehension cleanly, mapping each word to its length in one concise expression instead of a loop. `safe_divide()` correctly wraps the division in try/except, catching `ZeroDivisionError` and returning `None` as a sensible fallback—defensive coding that handles the edge case without crashing.

**Recommendations:**
- Add a few more test cases to the `if __name__ == "__main__"` block—edge cases like `word_lengths([])` (empty list) and `safe_divide(10, 0.0001)` (near-zero divisor). Testing the boundary strengthens confidence in the implementations.

**Streak:** 1 day — first exercise commit

---

## 2026-08-18

**Since yesterday:** No exercise commits — setup only

**What I saw:** This is the first real review after bootstrapping the coach earlier today. Everything in `git log` since the baseline (`28f7024`) is either the coach's own setup/cleanup commits — adding `.progress/` and `PROGRESS.md` at the repo root, then removing a stray copy that accidentally landed in `2students/.progress/` — or one unrelated commit (`7f61ae9d`, "Delete claude_practice directory") made via the GitHub web UI. Nothing touched `2students/` exercise content, and there's no `exercises/` folder there yet, so there's no student work to review.

**Recommendations:**
- Add your first exercise files under `2students/` (an `exercises/` folder works well) so tomorrow's review has real code to look at.

**Streak:** 0 days — exercise work hasn't started yet

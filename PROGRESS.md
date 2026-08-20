# Progress Log

This is the running log kept by the Student Progress Coach (see `2students/CLAUDE.md`). Each daily review adds one dated entry, prepended to the top of this file — newest first — so today's review is always the first thing you see. Entries are never edited or reordered after the fact.

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

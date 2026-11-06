# Presentation outline: Serie A Match Predictor

A 12-slide storyboard for a 10–15 minute talk (about 13 minutes at the suggested timings). It follows the [project story](project_story.md); the [visual guide](visual_guide.md) explains each chart and what to fix before using it. Every number below comes from the saved results; the full tables are in [methodology.md](methodology.md).

**Ground rules for the slides:** one message per slide, at most three short points, no code, no formulas beyond "lower is better", and no table larger than five rows. Numbers go in the speaker notes when they don't fit on the slide.

| # | Title | Visual | Time |
|---|---|---|---|
| 1 | Can thirty years of results predict a football match? | `docs/screenshots/app_predictions.png` | 0:45 |
| 2 | Home wins 45% of the time, and that's the bar to beat | `outputs/figures/02_outcome_distribution.png` | 1:00 |
| 3 | The game keeps changing | `outputs/figures/04_home_advantage_by_season.png` | 1:00 |
| 4 | No peeking: every feature comes from the past | `outputs/figures/10_form_difference_vs_result.png` | 1:15 |
| 5 | Three algorithms, one answer | `outputs/figures/21_model_agreement.png` | 1:00 |
| 6 | Testing like a forecaster, one season at a time | new diagram: walk-forward timeline | 1:30 |
| 7 | The bookmaker still wins | `outputs/figures/33_backtest_pooled_summary.png` | 1:30 |
| 8 | Two good ideas that didn't make the cut | `outputs/figures/43_elo_pooled_intervals.png` | 1:15 |
| 9 | We can estimate how many goals, not whether | `outputs/figures/51_over_under_2_5_vs_market.png` | 1:15 |
| 10 | The league's scoring drifts, and following it almost works | `outputs/figures/64_scoring_by_season.png` | 1:00 |
| 11 | An app that is honest about what it knows | `docs/screenshots/app_research.png` | 1:00 |
| 12 | What worked, what didn't, what we learned | text slide | 1:15 |

---

## Slide 1: Can thirty years of results predict a football match?

**Central message:** We built models that turn past results into home/draw/away probabilities, and tested them honestly against the bookmakers.

**On the slide:**
- 34 Serie A seasons, 11,826 matches
- Three machine-learning models, one bookmaker benchmark
- An interactive app that shows predictions and evidence side by side

**Visual:** `docs/screenshots/app_predictions.png`: the Match predictions tab, so the audience sees the end product first.

**Numbers:** 34 seasons (1993-94 to 2026-27 in progress); 11,826 completed matches; latest match 2 November 2026.

**Speaker notes:** Start with the product, then the question. "This is the app at the end of the project: pick two clubs and a date, and three models give you the probability of a home win, a draw or an away win. Today I want to tell you how we got there, and especially how we checked whether these numbers can be trusted." Make clear from the start that the goal is *probabilities*, not picking winners: a forecast that says 48% home win should be right about 48% of the time. Mention the data source (Football-Data.co.uk, free historical results and odds). Set expectations: the most interesting results of this project include things that *didn't* work, and the honest comparison with the bookmakers.

**Transition:** "Before building anything, we needed to understand what we were trying to predict."

---

## Slide 2: Home wins 45% of the time, and that's the bar to beat

**Central message:** Football results are dominated by home wins and low scores, which makes them hard to predict and sets a clear minimum standard.

**On the slide:**
- Home win 45%, draw 27%, away win 27%
- "Always predict a home win" is right 45% of the time
- Most matches end 1-1, 1-0 or 2-1

**Visual:** `outputs/figures/02_outcome_distribution.png`

**Numbers:** home 45.4%, draw 27.3%, away 27.3% over 11,826 matches; 2.66 goals per match on average; 1-1 is the most common score (12.5%).

**Speaker notes:** Point at the bars first: nearly half of all matches are home wins, so a model that ignores everything and always says "home win" is right 45% of the time. That is our baseline, and any model has to beat it. Then the heatmap: the darkest squares are 1-1, 1-0 and 2-1. When a single goal decides most matches, randomness plays a huge role, so nobody, including bookmakers, can be right most of the time. Draws are as frequent as away wins but have no favourite, so expect them to be the hardest outcome to forecast. Keep this slide quick: it is context.

**Transition:** "But these averages hide something important: the game has been changing."

---

## Slide 3: The game keeps changing

**Central message:** Home advantage has roughly halved in thirty years, so a model must always be tested on seasons after the ones it learned from.

**On the slide:**
- 1990s: home teams won about half their matches
- Last five seasons: about 40%; away wins up from 1 in 5 to 1 in 3
- Test on the future, never on the past

**Visual:** `outputs/figures/04_home_advantage_by_season.png` (crop to the top panel or enlarge the labels; see the visual guide).

**Numbers:** home goal edge about 0.58 goals per match in the 1990s, under 0.2 recently; points-per-game spread between clubs up from about 0.41 to about 0.50 (a more unequal league).

**Speaker notes:** Walk along the blue line from left to right: home wins drifting down from about 50% to about 40%, while the orange away-win line climbs. The grey band is the period without fans in 2020-21: home advantage was low then, but the decline had started long before and never came back. At the same time the league became more unequal, so *who* is playing now matters more than *where*. Two lessons for modelling: team-strength features should matter most, and evaluation must respect time. If we shuffled all matches and tested on a random sample, we would be testing on the past with knowledge of the future.

**Transition:** "So what can a model know about a match before kick-off?"

---

## Slide 4: No peeking: every feature comes from the past

**Central message:** We built 50 pre-match features from results strictly before each match, and tested that they never leak the future.

**On the slide:**
- 25 numbers per club: recent form, home/away form, this season, last season, promoted
- Built only from matches on earlier dates
- Tested: changing later results never changes earlier features

**Visual:** `outputs/figures/10_form_difference_vs_result.png`

**Numbers:** 50 features (25 per team); draws stay between about 19% and 31% whatever the form gap; 34% of matches lack last-season statistics (mostly promoted clubs).

**Speaker notes:** The chart shows the features work: as the home team's recent form improves relative to the visitor's, home wins rise from about 15% to about 75%. But look at the green draw line, flat between roughly 20% and 30%: form doesn't tell you when a draw is coming. Then explain *data leakage* with an example: if "goal difference this season" for a match included that match's own result, the model would be secretly reading the answer. It would look brilliant in testing and fail in real life. Our rule: only matches from earlier dates, not even other games on the same day. We have automated tests for this, including one that rewrites later results and checks that nothing earlier changes. Promoted clubs arrive with no Serie A history; we leave that visible rather than inventing numbers.

**Transition:** "With these features, we trained three very different kinds of model."

---

## Slide 5: Three algorithms, one answer

**Central message:** Logistic Regression, Random Forest and XGBoost give almost identical probabilities, so the algorithm is not the limiting factor.

**On the slide:**
- Home-win probabilities correlate at 0.97–0.98
- Same predicted outcome in about 88% of matches
- All three rely on team strength: last season and this season

**Visual:** `outputs/figures/21_model_agreement.png`

**Numbers:** validation 2025-26 accuracy 51.6–52.9% against 38.9% for the baseline; log loss 1.002–1.005 against 1.102.

**Speaker notes:** Each dot is one match from the 2025-26 season; the axes are two models' home-win probabilities. If the models disagreed, the cloud would be wide. Instead it hugs the diagonal. A linear model and two tree ensembles, which can find very complex patterns, end up saying the same thing. Briefly explain overfitting: with noisy data a flexible model can memorise noise, like a student memorising last year's exam; the best settings were the simple ones (strong regularisation, very shallow trees). The practical conclusion is important for the rest of the talk: if you want better predictions, you need better information, not a fancier algorithm. Note that 2025-26 was the season used to choose the settings, so these are not independent test scores; the next slide fixes that.

**Transition:** "So how do we judge them fairly, and how good are they really?"

---

## Slide 6: Testing like a forecaster, one season at a time

**Central message:** We replayed seven seasons as if living through them, and scored probabilities, not just picks.

**On the slide:**
- Each season predicted only from the seasons before it
- Scored with log loss: confident mistakes cost the most
- 2,657 matches, every forecast on exactly the same ones

**Visual:** a new walk-forward timeline diagram (proposed file `docs/diagrams/walk_forward_timeline.png`, not yet created; specification in the [visual guide](visual_guide.md#gaps-new-explanatory-diagrams)). Until it exists, use the text diagram in the [project story](project_story.md#5-judging-forecasts-fairly-chronological-walk-forward-testing).

**Numbers:** backtest seasons 2018-19 to 2024-25; log loss of a 50% call = 0.69, of a 10% call that comes true = 2.30; one third on every outcome = 1.099.

**Speaker notes:** Two ideas on this slide. First, *walk-forward testing*: for each season from 2018-19 to 2024-25, we hide everything after it, choose the model's settings using only earlier seasons, train, and predict. So every prediction was made with only the knowledge available at the time. Second, *log loss*: we score the probability the model gave to what actually happened. Say 50% and it happens: 0.69. Say 10% and it happens anyway: 2.30, a heavy penalty for overconfidence. Lower is better. Also mention *calibration*, the weather-forecaster test: of all the days with "40% chance of rain", it should rain on about 40%. Our models are reasonably calibrated. Finally, honesty: this is retrospective research, and these seven seasons were reused across several experiments, so later gains on them are exploratory.

**Transition:** "Now the real test: how do the models compare with people whose job is setting odds?"

---

## Slide 7: The bookmaker still wins

**Central message:** The models clearly beat the baseline, but the bookmaker's probabilities are better, and not by luck.

**On the slide:**
- Models: about 0.11 better than the baseline, in every season
- Bookmaker: better than every model, first in 6 of 7 seasons
- The gap is information (team news, transfers), not algorithms

**Visual:** `outputs/figures/33_backtest_pooled_summary.png` (use the log-loss panel only). Backup: `outputs/figures/32_backtest_gap_to_bookmaker.png` for questions about individual seasons.

**Numbers:** pooled log loss: baseline 1.092; Logistic Regression 0.983; Random Forest 0.983; XGBoost 0.981; bookmaker 0.960. Gap 0.021–0.023, 95% intervals about +0.013 to +0.031. Models' and bookmaker's home-win probabilities correlate at about 0.92–0.93.

**Speaker notes:** Read the chart from the right: the grey baseline at 1.092, then a big jump to the three models around 0.98, and the bookmaker further left at 0.960. Explain the bookmaker probabilities: odds converted to probabilities, with the bookmaker's built-in margin removed; used only for comparison, never as an input. The bookmaker is ahead in six of seven seasons, and the bootstrap interval for the gap excludes zero. In plain words, the bookmaker's lead is about a fifth of the distance the models travelled from the naive baseline. Where do they disagree most? Newly promoted clubs, end-of-season matches and big games: places where knowing squads, injuries and motivation matters. Pre-empt a question: on the 100 matches of 2026-27 the models scored slightly better (0.882 against 0.898), but 100 matches cannot separate them, so we do not claim the models beat the market. Draws remain unpredictable for everyone.

**Transition:** "If information is the gap, could better team-strength measures close it?"

---

## Slide 8: Two good ideas that didn't make the cut

**Central message:** Elo ratings and recency weighting both gave small gains that were not statistically confirmed, so neither is used.

**On the slide:**
- Elo: slightly better for every model, not confirmed after correction
- Recent seasons weighted more: no strategy beat equal weighting reliably
- Decision rules were written before each experiment

**Visual:** `outputs/figures/43_elo_pooled_intervals.png` (log-loss panel only). Optional hook: `outputs/figures/38_elo_club_histories.png` to show what Elo ratings look like.

**Numbers:** Elo lowered pooled log loss by 0.0014–0.0039; Random Forest's adjusted interval ended at +0.0001; Elo gap vs last-season points gap: rank correlation 0.87. Recency: best strategy (5-year half-life) −0.0027, better in 6 of 7 seasons, intervals reach zero; best of six strategies tried.

**Speaker notes:** Elo is the chess rating system: clubs gain points by beating others, more for surprising wins. Adding it to the features improved every model a little, which sounds convincing. But the rule we wrote *before* running it required a statistically clear gain after correcting for testing three models, and none passed: Random Forest missed by a hair. The reason is visible in the data: the Elo gap mostly repeats what last season's points already tell the model. Then recency: since the game changes, maybe recent seasons deserve more weight. Shorter memories were worse; a gentle 5-year half-life was best but within the noise, and it was the best of six attempts on the same seasons, which is exactly how luck masquerades as discovery. Both stay as candidates to test on new seasons. The point of this slide: knowing when *not* to ship an improvement.

**Transition:** "So 1X2 probabilities are close to the limit of our data. What about goals?"

---

## Slide 9: We can estimate how many goals, not whether

**Central message:** Models predict expected goals well, but their Over/Under and both-teams-to-score probabilities are no better than history and worse than the market.

**On the slide:**
- Expected goals: 10% better than the league average, every season
- Over/Under 2.5: about as good as a coin flip
- Probabilities too extreme, so none are shown as forecasts

**Visual:** `outputs/figures/51_over_under_2_5_vs_market.png`

**Numbers:** Poisson deviance 2.147 (Poisson regression) against 2.394 (league average); Over 2.5 log loss 0.6912 (best Poisson) against 0.6945 (league average) and 0.6786 (market); coin flip = 0.693; both teams scored in 55.3% of backtest matches, under-predicted by 3.8 points.

**Speaker notes:** Goals in football follow a classic statistical pattern, the Poisson distribution, so you can estimate each side's expected goals and derive any goal probability from them. Our Poisson regression estimates expected goals clearly better than the league average, in all seven seasons. But turn that into "will there be more than 2.5 goals?" and it falls apart. The grey line is a coin flip; our models hover around it, while the green market line sits clearly below. The probabilities were also too extreme: when the model said 70%, it happened less often. Estimating the average is easier than getting yes/no probabilities right. Because the models failed their readiness rule, the app shows only *historical* goal frequencies, clearly labelled as not forecasts.

**Transition:** "We looked for the reason, and found it in the league itself."

---

## Slide 10: The league's scoring drifts, and following it almost works

**Central message:** Scoring levels shift between seasons; following recent league scoring gave our best goal probabilities yet, but it is promising, not validated.

**On the slide:**
- Goals per match: 2.68 → 3.05 → about 2.6 in five seasons
- Following the last 200 matches gave the best scores so far
- Passed 4 of 5 criteria for Over 2.5; failed calibration

**Visual:** `outputs/figures/64_scoring_by_season.png`

**Numbers:** 2.68 goals per match (2018-19), 3.04 and 3.05 (2019-20, 2020-21), about 2.56–2.61 from 2022-23. Adaptive frequency vs static frequency on Over 2.5: −0.0060 log loss, interval −0.0099 to −0.0023 after correcting for 8 comparisons, better in 6 of 7 seasons, calibration slope 1.44 (too timid). Market still better: 0.6786 against 0.6886.

**Speaker notes:** The dashed lines are the long-run averages the static methods assume; the solid lines are what actually happened. Away scoring, in orange, has been above its long-run average in every recent season: a structural change, not noise. So we tried methods that follow the last 200 league matches, pulled gently toward history. For Over 2.5 this was the only comparison in its family that stayed clear of zero after correction, and it was better in six of seven seasons. But it adjusted too little: right direction, too timid, so it failed the calibration criterion, and we did not ship it. We also did not "fix" its settings afterwards, because that would be tuning on the test. The honest status: our most promising goals idea, waiting for seasons it has never seen.

**Transition:** "All of this evidence ends up in the app, and the app is built to be honest about it."

---

## Slide 11: An app that is honest about what it knows

**Central message:** The app separates model predictions, historical statistics and research evidence, and warns when its data is out of date.

**On the slide:**
- Three tabs: predictions / goals history / research
- Warns when data is stale or a fixture is far ahead
- No unvalidated forecasts; never retrains anything

**Visual:** `docs/screenshots/app_research.png` (or `docs/screenshots/app_goals.png` to show the "descriptive history, not a forecast" label).

**Numbers:** stale-data warning after 10 days; 430 automated tests (413 run in a fresh clone without data); the research tab reads the saved result files directly.

**Speaker notes:** Give a short live demo or walk through the screenshot. The Match predictions tab shows each model's probabilities and where they disagree. The Goals tab shows each club's record, explicitly labelled as history, not forecasts. The Research tab shows the evidence from this talk, read straight from the saved results, including why Elo and the goals models are *not* used. Mention the safeguards: the latest-data date is always visible; a warning appears when the data is more than 10 days old or the fixture is far ahead; the historical demo mode rebuilds any past match as it looked that morning and warns when that season was used for training. Behind it, 430 automated tests, and a reproducibility check: rebuilding everything from a fresh copy reproduced the published results.

**Transition:** "So, what did thirty years of football teach us?"

---

## Slide 12: What worked, what didn't, what we learned

**Central message:** The models are useful and honestly tested, the bookmaker is better, and the biggest lesson is that information and evaluation matter more than algorithms.

**On the slide (three columns):**
- **Worked:** leakage-safe features; models beat the baseline every season; expected goals
- **Didn't:** beating the bookmaker; predicting draws; goal-event probabilities
- **Learned:** information beats algorithms; seasons vary more than models; write the rule before you look

**Visual:** none: a clean three-column text slide. Optionally reuse the log-loss panel of `outputs/figures/33_backtest_pooled_summary.png` as a small reminder.

**Numbers:** about 0.11 better than the baseline; about 0.02 behind the bookmaker; differences between models at most 0.003.

**Speaker notes:** Summarise in three sentences. What worked: a clean, leakage-safe pipeline and models that beat the naive baseline in every season, plus a testing framework that makes the next idea cheap to test and hard to fool ourselves with. What didn't: we did not beat the bookmaker, nobody can predict draws, and none of the goal-event models were good enough to ship. What we learned: the gap is information, not algorithms; one season can make any model look good or bad; and deciding in advance what counts as success protects you from false discoveries. Close with next steps: test the leading candidates (Random Forest with Elo, the adaptive goal frequency) on seasons they have never seen, and add new information such as line-ups. Invite questions.

**Transition:** Questions.

---

## Backup slides (for questions)

| Likely question | Visual | Key numbers |
|---|---|---|
| "Did the models beat the bookmaker on 2026-27?" | `outputs/figures/24_probability_scores.png` | Test log loss 0.882 (LR, XGB), 0.895 (RF), 0.898 (bookmaker) on only 100 matches; intervals far too wide to separate them |
| "Are the probabilities calibrated?" | `outputs/figures/26_reliability_validation.png` (home-win row) | Reasonably calibrated; slightly over-confident about strong home favourites |
| "Season by season?" | `outputs/figures/32_backtest_gap_to_bookmaker.png` | Bookmaker better in 6 of 7 seasons; LR first in 2022-23 by 0.001 |
| "What do the features look like?" | `outputs/figures/11_feature_signal.png` | Season and last-season goal difference correlate most with the result (about 0.37) |
| "How does the adaptive method compare to everything?" | `outputs/figures/71_drift_decision_intervals.png` | Only one of the eight comparisons excludes zero after correction |

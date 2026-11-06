# Visual guide

An audit of the project's 71 figures (`outputs/figures/`) and three app screenshots (`docs/screenshots/`), with the visuals recommended for the [project story](project_story.md) and the [presentation](presentation_outline.md).

**Do not overwrite the originals.** The figures in `outputs/figures/` are experiment outputs and are rewritten when a notebook runs. Save any cropped or relabelled version for slides under a new name in a separate folder (suggested: `docs/presentation_figures/`, e.g. `33_backtest_pooled_summary_logloss_only.png`), and keep every number exactly as in the original.

## Recommended visuals

| # | File | What the viewer should understand | Use directly? | Improvements for a slide |
|---|---|---|---|---|
| 1 | `docs/screenshots/app_predictions.png` | The end product: three models give home/draw/away probabilities for any fixture | Yes | Crop to the sidebar and the three model cards; at 1000 px wide the small text is hard to read on a projector |
| 2 | `outputs/figures/02_outcome_distribution.png` | Home wins 45%, draws and away wins 27% each; most matches are low-scoring | Yes | Fine as is; for a short talk the left bar chart alone is enough |
| 3 | `outputs/figures/04_home_advantage_by_season.png` | Home advantage has roughly halved since the 1990s | With changes | Tall two-panel figure with small season labels: use the top panel only, or enlarge the fonts; add a one-line takeaway ("home wins ~50% → ~40%") |
| 4 | `outputs/figures/10_form_difference_vs_result.png` | Better recent form means more wins, but draws stay flat | Yes | The x-axis label ("points per game ... over the last 10 matches") is technical; add an annotation such as "home team in much better form →" |
| 5 | `outputs/figures/21_model_agreement.png` | The three models give almost the same probabilities (correlation 0.97–0.98) | Yes | One panel (XGBoost vs Logistic Regression) is enough; enlarge the correlation label |
| 6 | `outputs/figures/33_backtest_pooled_summary.png` | Models beat the baseline; the bookmaker beats the models | With changes | Use the log-loss panel only; replace "LR / RF / XGB" with full names; add "lower is better" in large text; consider highlighting the bookmaker row |
| 7 | `outputs/figures/32_backtest_gap_to_bookmaker.png` | The bookmaker is ahead in almost every season, not just on average | Backup | Error bars need explaining; the title "Model log loss minus bookmaker log loss" is technical; label the zero line "no difference" |
| 8 | `outputs/figures/43_elo_pooled_intervals.png` | Elo helps a little but the gain is not confirmed | With changes | Use the log-loss panel only; the thick/thin bar legend sits in a small footnote: enlarge it or annotate "95% interval" and "corrected for 3 models" directly on the Random Forest bar |
| 9 | `outputs/figures/38_elo_club_histories.png` | What Elo ratings look like over 30 years | Optional | Six dense panels; one or two clubs are enough for a hook. Gaps in a line are seasons outside Serie A |
| 10 | `outputs/figures/36_recency_pooled_intervals.png` | No recency strategy beats equal weighting reliably | Story only | Two panels and six strategies are a lot for a slide; the project story uses it, and the presentation mentions the result in words |
| 11 | `outputs/figures/51_over_under_2_5_vs_market.png` | Goal-event models are close to a coin flip; the market is clearly better | Yes, with a caveat | The y-axis spans only 0.665–0.71, which makes small differences look large: say so, or label the coin-flip line in larger text |
| 12 | `outputs/figures/64_scoring_by_season.png` | Scoring levels drift; the static methods assume an outdated average | Yes | The four-line legend is busy: label the lines directly or fade the pre-2018 seasons |
| 13 | `docs/screenshots/app_research.png` | The app shows the evidence, including what is *not* deployed | Yes | Crop to the key findings and the decisions table |

The project story also uses `outputs/figures/32_backtest_gap_to_bookmaker.png` and `outputs/figures/36_recency_pooled_intervals.png`, where the reader has time to study them.

## Colour consistency

Colours are consistent within the notebooks, but not between the notebooks and the app:

| | Notebook figures | App and screenshots |
|---|---|---|
| Logistic Regression / Random Forest / XGBoost | blue / orange / green | green / purple / pink |
| Home win / draw / away win | blue / green / orange | blue / grey / orange |

If slides mix figures and screenshots, either say so or recolour the figures you edit to one palette. Home blue and away orange already match everywhere.

## Figures considered but not recommended for slides

| File | Why not on a slide | Where it is useful |
|---|---|---|
| `outputs/figures/11_feature_signal.png` | Raw feature names (`venue_form10_gd`) mean nothing to a general audience | Backup slide or the methodology |
| `outputs/figures/26_reliability_validation.png` | Twelve small panels; calibration is easier to explain in words | Backup: show the home-win row only |
| `outputs/figures/31_backtest_metrics_by_season.png` | Four metrics in one figure; 33 makes the same point more simply | README results section |
| `outputs/figures/24_probability_scores.png` | Validation and test side by side invites over-reading the 100-match test | Backup for "did you beat the bookmaker on 2026-27?" |
| `outputs/figures/53_goals_reliability.png` | Five small panels | Crop the Over 2.5 panel to illustrate "too extreme" probabilities |
| `outputs/figures/71_drift_decision_intervals.png` | Eight comparisons with two interval types | Backup for statistically minded audiences |

## Gaps: new explanatory diagrams

These concepts are central to the story but have no figure. Each would be a simple diagram, built from facts already in the documentation; none needs new analysis. Suggested files go in `docs/diagrams/` (not created yet).

1. **Walk-forward timeline** (`docs/diagrams/walk_forward_timeline.png`): the most important missing visual (presentation slide 6).
   - **Content:** seasons on the horizontal axis, one row per evaluation season from 2018-19 to 2024-25. Each row shows:
     - training seasons from 1993-94 to S-2;
     - S-1 marked as "choose settings";
     - S marked as "predict";
     - later seasons greyed out as "hidden".
   - **Extra bands:** 2025-26 as "validation and tuning (saved models)" and 2026-27 as "held-out test, 100 matches".
2. **"What the model knows on match day"** (`docs/diagrams/leakage_timeline.png`).
   - **Content:** a single fixture on a timeline. Earlier matches are included, and the same-day matches and the fixture itself are excluded.
   - **Purpose:** explains data leakage in one glance (slide 4).
3. **Log loss intuition** (`docs/diagrams/log_loss_curve.png`).
   - **Content:** the curve −log(p) for p from 0.05 to 1, with three marked points:
     - 50% → 0.69;
     - 10% → 2.30;
     - one-third on each outcome → 1.10.
   - **Purpose:** shows why confident mistakes are expensive (slide 6). It is a mathematical curve, so it involves no results.
4. **Pipeline overview** (`docs/diagrams/pipeline.png`).
   - **Content:** Football-Data CSVs → cleaned matches → 50 features → three models → walk-forward evaluation and bookmaker benchmark → Streamlit app.
   - **Purpose:** useful for the README and as an opening or closing slide.
5. **Decision scorecard** (`docs/diagrams/decision_scorecard.png`).
   - **Content:** one row per experiment (Elo, recency, Poisson goals, direct/recalibrated, adaptive rates), with the decision-rule criteria as ticks and crosses and the final "deployed?" column.
   - **Source:** the results in [methodology.md](methodology.md); every row ends in "no".
   - **Purpose:** summarises slides 8–10 in one picture.

A text version of the walk-forward diagram is already in the [project story](project_story.md#5-judging-forecasts-fairly-chronological-walk-forward-testing).

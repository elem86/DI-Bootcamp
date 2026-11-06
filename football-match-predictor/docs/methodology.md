# Methodology and detailed results

The detailed write-up behind the [README](../README.md): every pipeline step, model and experiment, with its command, design, decision rule and full results. All numbers are the published results, computed from data downloaded on 5 November 2026, whose latest completed match was played on 2 November 2026. How to reproduce them is described in the [README](../README.md#reproducing-the-research).

The project was built as a sequence of commits, and the text refers to some of them by number:

| Commit | Step |
|---|---|
| 5 | Leakage-safe pre-match features |
| 6 | Logistic Regression baseline |
| 7 | Random Forest and XGBoost; the saved models' settings, chosen on 2025-26 |
| 10 | Walk-forward backtest, 2018-19 to 2024-25 |
| 15 | Poisson goals model |
| 16 | Direct Over/Under and BTTS experiment |
| 17 | Adaptive scoring rates experiment |

## Contents

- [Data](#data)
- [Features](#features)
- [Baseline model: Logistic Regression](#baseline-model-logistic-regression)
- [Tree models: Random Forest and XGBoost](#tree-models-random-forest-and-xgboost)
- [Probability calibration and bookmaker benchmark](#probability-calibration-and-bookmaker-benchmark)
- [Excel workbook](#excel-workbook)
- [Walk-forward backtesting](#walk-forward-backtesting)
- [Recency weighting experiment](#recency-weighting-experiment)
- [Interactive prediction app](#interactive-prediction-app)
- [Goals, Over/Under and BTTS analysis](#goals-overunder-and-btts-analysis)
- [Elo rating experiment](#elo-rating-experiment)
- [Poisson goals model](#poisson-goals-model)
- [Direct Over/Under and BTTS experiment](#direct-overunder-and-btts-experiment)
- [Adaptive scoring rates experiment](#adaptive-scoring-rates-experiment)

## Data

Historical match data comes from [Football-Data.co.uk](https://www.football-data.co.uk/italym.php) (Serie A results and match statistics by season). Raw files are downloaded by the project scripts rather than committed to the repository.

Download every Serie A season (1993-94 to the current season) into `data/raw/`:

```powershell
python -m src.data_loader                      # all seasons
python -m src.data_loader --from-season 2005   # 2005-06 onwards only
python -m src.data_loader --force              # re-download everything
```

Files are saved as `data/raw/I1_<season>.csv` (e.g. `I1_2425.csv` for 2024-25; `I1` is Football-Data's code for Serie A). Re-running the script skips seasons already downloaded but always refreshes the current season, which is updated after every matchday.

Then combine all seasons into one standardised dataset:

```powershell
python -m src.preprocessing
```

This reads every raw file, fixes the format differences between seasons (ragged rows, two date formats, renamed odds columns, byte-order marks), validates the result, and writes `data/processed/matches.csv`, one row per match. Load it in notebooks with `src.preprocessing.load_matches()` to keep the correct column types. [notebooks/01_data_exploration.ipynb](../notebooks/01_data_exploration.ipynb) walks through each step.

| Columns | Content | Available from |
|---|---|---|
| `season`, `season_start`, `date`, `time` | When the match was played (`time` from 2019-20) | 1993-94 |
| `home_team`, `away_team` | Team names | 1993-94 |
| `home_goals`, `away_goals`, `result` | Full-time score and result (`H` / `D` / `A`) | 1993-94 |
| `ht_home_goals`, `ht_away_goals`, `ht_result` | Half-time score and result | 1995-96 |
| `home_shots`, `home_shots_on_target`, `home_fouls`, `home_corners`, `home_yellow_cards`, `home_red_cards` (and `away_` equivalents) | Match statistics | 2005-06 |
| `b365_home`, `b365_draw`, `b365_away` | Bet365 decimal odds | 2002-03 |
| `avg_odds_home`, `avg_odds_draw`, `avg_odds_away` | Market-average decimal odds | 2005-06 |
| `referee` | Referee | 2005-06 to 2006-07 only |

## Features

Build pre-match team performance features from the processed matches:

```powershell
python -m src.feature_engineering
```

This writes `data/processed/features.csv`: one row per match with identifiers, the prediction targets (`result`, plus `home_goals` and `away_goals`, copied unchanged) and 50 features, 25 for the home team (`home_…`) and 25 for the away team (`away_…`). Load it with `src.feature_engineering.load_features()`. [notebooks/02_feature_analysis.ipynb](../notebooks/02_feature_analysis.ipynb) explores the features.

| Feature family | Columns (per team) | Window |
|---|---|---|
| History length | `history_matches` | Earlier matches in the team's current Serie A spell |
| Recent form | `form5_*`, `form10_*` | Last 5 / 10 matches, any venue |
| Venue form | `venue_form5_*`, `venue_form10_*` | Last 5 / 10 home matches (home team) or away matches (away team) |
| Season to date | `season_played`, `season_ppg`, `season_gf`, `season_ga`, `season_gd` | Earlier matches this season |
| Last season | `prev_season_ppg`, `prev_season_gd`, `promoted` | The whole previous season |

Form features are per-game averages: `ppg` (points: 3 for a win, 1 for a draw), `gf` (goals scored), `ga` (goals conceded) and `gd` (goal difference).

- **No leakage:** every feature uses only matches from strictly earlier dates. A match never sees its own score, and matches on the same day never see each other's results. Unit tests check this, including a test that changes later results and confirms earlier features stay the same.
- **Season transitions:** form windows carry over the summer break for teams that stay in Serie A. A team arriving after a gap (promoted) starts with no history, because its Serie B matches are not in the data. Season-to-date features restart every season.
- **Missing history is explicit:** a form window is missing (NaN) until the team has played the full number of matches, rather than averaged over fewer. Last-season features are missing for promoted teams, which the `promoted` flag marks. In 1993-94, the first season in the data, `promoted` is unknown.

## Baseline model: Logistic Regression

Train, tune and evaluate the first model (after building the features):

```powershell
python -m src.train --models logistic_regression
```

Multinomial Logistic Regression predicts home win (`H`), draw (`D`) or away win (`A`) from the pre-match features, using a strictly chronological split:

| Step | Seasons |
|---|---|
| Train | 1993-94 to 2024-25 |
| Validate: choose the feature set and regularisation strength `C` by log loss | 2025-26 |
| Final training: refit the chosen model | 1993-94 to 2025-26 |
| Test | 2026-27, completed matches only |

The model is a scikit-learn `Pipeline`, so preprocessing is fitted on the training rows only:
1. `SimpleImputer(strategy="median", add_indicator=True)` fills missing history (no matches are dropped) and adds a 0/1 indicator for each feature that can be missing.
2. `StandardScaler` puts every feature on the same scale.
3. L2-regularised `LogisticRegression` makes the prediction.

The `promoted` flag is kept as a feature. Two feature sets are compared: all 50 features, and a compact set of 22 that drops redundant columns (`gd = gf − ga`, and the 5-match windows). The baseline always predicts the most frequent training outcome (home win), with the training outcome frequencies as its probabilities.

Outputs (git-ignored):
- `outputs/models/logistic_regression.joblib`: the fitted pipeline and its feature list
- `outputs/models/logistic_regression_metrics.json`: all metrics and tuning results
- `outputs/predictions/logistic_regression_test_2026-27.csv`: one row per test match, with date, teams, actual and predicted outcome, and the home-win, draw and away-win probabilities (`…_validation_2025-26.csv` holds the same for the validation season)

[notebooks/03_baseline_model.ipynb](../notebooks/03_baseline_model.ipynb) explains the process and visualises the results.

**Results.** The selected model uses the compact feature set with strong regularisation (`C = 0.001`).

| Season | Model | Accuracy | Macro F1 | Log loss | Home win acc. | Draw acc. | Away win acc. |
|---|---|---|---|---|---|---|---|
| Validation 2025-26 (380 matches) | Logistic Regression | 51.8% | 0.413 | 1.002 | 79.1% | 3.0% | 57.9% |
| | Baseline | 38.9% | 0.187 | 1.102 | 100% | 0% | 0% |
| Test 2026-27 (100 matches) | Logistic Regression | 60.0% | 0.460 | 0.882 | 80.9% | 6.2% | 56.8% |
| | Baseline | 47.0% | 0.213 | 1.060 | 100% | 0% | 0% |

- The model beats the baseline on accuracy, macro F1 and log loss in both seasons.
- The test season has only 100 matches (95% interval for accuracy about ±10 points), so the 380-match validation season is the steadier estimate.
- Draws are almost never predicted: draw probabilities stay between about 0.12 and 0.40, so a draw is rarely the single most likely outcome. The draw probabilities are still sensible on average.
- Team strength drives the predictions, mainly last-season and season-to-date goal difference and points, and 10-match form.

## Tree models: Random Forest and XGBoost

Train all three models (Logistic Regression, Random Forest, XGBoost) and compare them:

```powershell
python -m src.train                                  # all models
python -m src.train --models random_forest xgboost   # selected models
```

The tree models use exactly the same features, chronological splits, tuning-by-validation-log-loss and metrics as the Logistic Regression baseline. Model definitions and tuning grids live in `src/models.py`.

| Model | Pipeline | Tuned on 2025-26 | Selected |
|---|---|---|---|
| Random Forest | Median imputer with missing indicators → `RandomForestClassifier` (400 trees) | `max_depth` × `min_samples_leaf` × feature set | compact, depth 12, 10 matches per leaf |
| XGBoost | `XGBClassifier` (learning rate 0.03); missing values handled natively | `max_depth` × `n_estimators` × feature set | compact, depth 2, 250 trees |

Each model saves its fitted pipeline, metrics and predictions under the same naming scheme as the Logistic Regression files (`outputs/models/<model>.joblib`, `outputs/predictions/<model>_test_2026-27.csv`, …). `outputs/models/model_comparison.csv` holds the side-by-side metrics. [notebooks/04_random_forest_xgboost.ipynb](../notebooks/04_random_forest_xgboost.ipynb) covers the tuning, the results, how far the models agree, and permutation feature importance.

| Season | Model | Accuracy | Macro F1 | Log loss |
|---|---|---|---|---|
| Validation 2025-26 (380 matches) | Logistic Regression | 51.8% | 0.413 | **1.002** |
| | Random Forest | **52.9%** | **0.430** | 1.003 |
| | XGBoost | 51.6% | 0.426 | 1.005 |
| | Baseline | 38.9% | 0.187 | 1.102 |
| Test 2026-27 (100 matches) | Logistic Regression | 60.0% | 0.460 | **0.882** |
| | Random Forest | **63.0%** | **0.506** | 0.895 |
| | XGBoost | 61.0% | 0.467 | **0.882** |
| | Baseline | 47.0% | 0.213 | 1.060 |

- All three models beat the baseline by a similar, large margin, but none clearly beats the others. Gaps of 1–3 accuracy points (1 point is a single test match) and at most 0.013 log loss are within the noise of these sample sizes.
- The models agree on the predicted outcome for about 88% of validation matches, and their home-win probabilities correlate at 0.97–0.98. The tree models find no extra non-linear patterns in these features, so the information in the features, not the choice of model, limits accuracy.
- Both tree models rely mainly on last-season and season-to-date strength, the same features that dominate the Logistic Regression.
- Draws are still rarely predicted, so this is a property of the problem rather than of the linear model.

## Probability calibration and bookmaker benchmark

[notebooks/05_probability_analysis.ipynb](../notebooks/05_probability_analysis.ipynb) checks how good the models' **probabilities** are, not just their top picks, and compares them with Bet365's betting odds. Nothing is retrained or tuned: the notebook reads the predictions saved by `python -m src.train`.

- **Prediction accuracy** (accuracy, macro F1) only looks at the most likely outcome.
- **Probability quality** looks at all three probabilities:
  - **Log loss**
  - **Multiclass Brier score**: the squared distance between the probabilities and the outcome; 0 is perfect, 0.667 is 1/3 each, 2 is certain and wrong.
  - **Calibration**: do the "40%" predictions happen about 40% of the time?

**Bookmaker benchmark.** Bet365's 1X2 odds (`b365_home/draw/away`) become probabilities as `1 / odds`, normalised so that each match's three probabilities sum to 1. This removes the bookmaker's margin (overround), which averages 5.5% in 2025-26 and 2026-27.
- The odds are used **only as a benchmark**, never as model features.
- They are matched to each fixture by date, home team and away team, and the stored result must agree.
- All 380 validation and 100 test matches have complete odds, so every forecast is scored on exactly the same fixtures.
- There is no betting-profit analysis.

Reusable code:
- `src/evaluate.py`: Brier score, per-match losses, reliability tables with Wilson intervals, per-outcome calibration summary, paired bootstrap.
- `src/bookmaker.py`: implied probabilities, overround, odds coverage, fixture matching.

| Season | Forecast | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Validation 2025-26 (380 matches) | Logistic Regression | 51.8% | 1.002 | 0.599 |
| | Random Forest | 52.9% | 1.003 | 0.601 |
| | XGBoost | 51.6% | 1.005 | 0.602 |
| | Bookmaker (Bet365) | **53.9%** | **0.982** | **0.584** |
| | Baseline | 38.9% | 1.102 | 0.669 |
| Test 2026-27 (100 matches) | Logistic Regression | 60.0% | **0.882** | 0.508 |
| | Random Forest | **63.0%** | 0.895 | 0.518 |
| | XGBoost | 61.0% | **0.882** | **0.508** |
| | Bookmaker (Bet365) | **63.0%** | 0.898 | 0.520 |
| | Baseline | 47.0% | 1.060 | 0.640 |

Paired bootstrap of the difference in log loss (10,000 resamples of the matches, 95% intervals):

| Comparison | Validation 2025-26 | Test 2026-27 |
|---|---|---|
| Logistic Regression − Bookmaker | +0.020 [+0.000, +0.040] | −0.016 [−0.054, +0.023] |
| Random Forest − Bookmaker | +0.021 [+0.002, +0.040] | −0.002 [−0.037, +0.033] |
| XGBoost − Bookmaker | +0.023 [+0.003, +0.042] | −0.015 [−0.048, +0.017] |
| Between the three models | every interval includes 0 | every interval includes 0 |

- **Validation:** the bookmaker's probabilities are better than every model's by about 0.02 log loss. The intervals lie only just above 0, so this is probably a real but small gap.
- **Test:** the models score slightly better, but 100 matches cannot separate any of the forecasts.
- **Accuracy is not probability quality:** on test, the bookmaker ties for the best accuracy yet has the worst log loss of the four.
- **Calibration:** the models are reasonably calibrated for home win, draw and away win. They are slightly over-confident about strong home favourites. Their largest misses are season-wide and shared with the bookmaker: fewer home wins than usual in 2025-26, and fewer draws so far in 2026-27.
- **Spread:** all four forecasts spread their probabilities almost equally. The bookmaker's edge comes from ranking matches better (team news, transfers, promoted teams), not from bolder probabilities. Nobody can single out likely draws.
- **Limitations:**
  - The bootstrap assumes independent matches and makes six comparisons per season without correction, so it is not a formal significance test.
  - Each split is a single season.
  - The models' settings were chosen on the validation season.

## Excel workbook

Generate a professional Excel workbook from the processed data and the saved predictions:

```powershell
python -m src.train          # only if outputs/predictions/ is empty (it is not tracked in git)
python -m src.excel_report   # writes outputs/excel/serie_a_prediction_analysis.xlsx
```

Then open `outputs/excel/serie_a_prediction_analysis.xlsx` in Microsoft Excel. If Excel opens it in Protected View, click **Enable Editing** so that the formulas calculate. The workbook is generated output, so it stays in the git-ignored `outputs/excel/` folder; only the generator (`src/excel_report.py`) is committed.

No model is retrained and no prediction changes: the generator reads `data/processed/matches.csv` and the saved per-match predictions in `outputs/predictions/`.

| Sheet | Contents |
|---|---|
| **Overview** | Dashboard with headline numbers (Excel formulas), the evaluation plan with sample sizes, key findings, metric definitions and charts |
| **Historical Trends** | One row per season: outcome shares, goals per match, odds coverage and bookmaker margin, all as `COUNTIFS`/`SUMIFS` formulas over Match Data; the in-progress season is highlighted |
| **Model Comparison** | Accuracy, macro F1, log loss, Brier score and per-outcome accuracy for each model, the bookmaker and the baseline, on validation and test; plus log-loss rank, gain over the baseline, and the baseline probabilities rebuilt from the match data |
| **Validation 2025-26** | 380 fixtures: each model's probabilities, plus formulas for its pick, whether it was correct, per-match log loss and Brier score |
| **Test 2026-27** | The same for the 100 completed matches; marked as an incomplete season |
| **Model vs Bookmaker** | Every fixture with Bet365 odds: overround and margin-free bookmaker probabilities (formulas), each forecast's per-match log loss, a season summary, and the paired-bootstrap intervals |
| **Match Data** | All 11,826 completed matches with results and odds (the source of the formulas) |

How to read it:
- **Colours:** blue text = values from the Python pipeline; black = Excel formulas; green = formulas that pull from another sheet.
- **Tables:** every table is an Excel table with filters, frozen headers and conditional formatting.
- **Check column:** the **Check (recomputed in Excel)** column on Model Comparison confirms that Excel's own formulas reproduce the Python metrics; every row should read `OK`.
- **Validation and test:** each has its own sheet, every table shows the number of matches, and the 2026-27 season is labelled *in progress* everywhere.

`tests/test_excel_report.py` checks the sheet names, that every fixture and probability matches the saved predictions, that the metrics match an independent evaluation, that the odds belong to the right fixtures, and that the formulas point at the right cells.

## Walk-forward backtesting

One validation season is a small sample, so `src/backtest.py` evaluates the same models on every season from 2018-19 to 2024-25, each predicted only from the seasons before it:

```powershell
python -m src.backtest                                   # all models, about 6 minutes
python -m src.backtest --models logistic_regression      # selected models
```

Results are saved to `outputs/backtest/` (git-ignored). [notebooks/06_walk_forward_backtesting.ipynb](../notebooks/06_walk_forward_backtesting.ipynb) runs the backtest and presents it.

**One fold per evaluation season S:**

1. Hide every season after S.
2. Tune the feature set and hyperparameters by fitting on seasons up to S-2 and scoring on S-1.
3. Refit the winner on all seasons up to S-1.
4. Predict S.

Season S's results are used only for scoring. The training window expands with every fold.

**Why each fold tunes its own settings:** the commit-7 settings were chosen on 2025-26, so reusing them for earlier seasons would let a later season configure an earlier forecast. Each fold instead selects from the same predefined grids using only past seasons. The grid *ranges* themselves were set in commit 7, which is documented as a small remaining caveat.

**Untouched:** the 2026-27 holdout (the code refuses it) and the 2025-26 validation results.

All forecasts are scored on identical fixtures with complete Bet365 odds: 2,657 of 2,660 matches (99.9%; 2020-21 and 2021-22 each have 1–2 matches without odds).

| Forecast | Pooled log loss | Pooled Brier | Pooled accuracy | Best / worst season log loss | Seasons ranked 1st |
|---|---|---|---|---|---|
| Logistic Regression | 0.983 | 0.585 | 53.5% | 0.949 / 1.005 | 1 |
| Random Forest | 0.983 | 0.585 | 53.4% | 0.947 / 0.999 | 0 |
| XGBoost | 0.981 | 0.584 | 54.0% | 0.940 / 0.997 | 0 |
| Bookmaker (Bet365) | **0.960** | **0.570** | **54.4%** | 0.934 / 0.983 | 6 |
| Baseline | 1.092 | 0.662 | 41.2% | 1.078 / 1.106 | 0 |

In 2022-23, the one season where the bookmaker isn't first, Logistic Regression ranks first. Its gap to the bookmaker there is −0.001, within the noise.

- **Two averages:** the backtest reports the **season mean** (each season counts once) and the **pooled** score over all 2,657 per-match predictions (each match counts once). With 378–380 scored matches per season they agree closely; macro F1 differs slightly, because it has to be computed from the pooled predictions.
- **Models vs bookmaker:** the bookmaker is first in 6 of 7 seasons. Pooled, every model trails it by 0.021–0.023 log loss, and the 95% paired-bootstrap intervals (about +0.013 to +0.031) exclude 0.
- **Between the models:** no model beats another. Each is best among the three in 2–3 seasons, and the pooled differences (≤ 0.003) have intervals that include 0.
- **Seasons vary more than models:** each model's log loss moves by about 0.05–0.06 between seasons. The commit-7 validation season 2025-26 was a hard one for the models *and* the bookmaker, and the gap to the bookmaker seen there (about 0.02) is typical.
- **Selected settings vary from fold to fold** (e.g. Logistic Regression `C` from 0.0003 to 1.0), but the scores barely change: the tuning landscape is flat.

## Recency weighting experiment

Does Logistic Regression predict better if recent seasons count more than old ones? `src/recency.py` adds three training strategies to the walk-forward backtest:

```powershell
python -m src.recency      # 7 strategies x 7 seasons, about 4 minutes
```

Results are saved to `outputs/recency/` (git-ignored). [notebooks/07_recency_experiments.ipynb](../notebooks/07_recency_experiments.ipynb) runs the experiment and presents it.

| Strategy | Training data | Settings tested |
|---|---|---|
| Expanding (control) | Every earlier season, equal weights (the commit-10 procedure) | — |
| Rolling window | Only the last *k* training seasons | *k* = 3, 5, 10 |
| Exponential decay | Every earlier season, older matches weighted less | half-life *h* = 1, 2, 5 years |

**Weight formula.** `age_i = (cutoff − date_i)` in days / 365.25, where `cutoff` is the date of the most recent training match. Then `w_i = 0.5 ** (age_i / h)`, rescaled so that the mean weight is 1:
- the total weight equals the number of matches, as in the control;
- a match *h* years older counts half as much.

**How each fold is run:**
- **Same procedure for every strategy:** each runs the full commit-10 fold procedure (past-only choice of feature set and `C`, then refit), with the strategy applied to both the inner tuning fits and the final fit.
- **Where weights apply:** weights go to the Logistic Regression step; the imputer and scaler are fitted on the same rows without weights.
- **Fresh control:** the control is re-evaluated with the same code, and reproduces the commit-10 Logistic Regression predictions exactly.
- **Identical fixtures:** all strategies are scored on the same 2,657 fixtures.
- **Untouched:** neither 2025-26 nor 2026-27 is used.

| Strategy | Pooled log loss | vs control | Seasons better than control | Season-level 95% interval |
|---|---|---|---|---|
| Expanding (control) | 0.9834 | — | — | — |
| Window: 3 seasons | 0.9850 | +0.0016 | 3 of 7 | −0.0085 to +0.0117 |
| Window: 5 seasons | 0.9848 | +0.0014 | 3 of 7 | −0.0083 to +0.0111 |
| Window: 10 seasons | 0.9822 | −0.0012 | 4 of 7 | −0.0067 to +0.0043 |
| Half-life: 1 year | 0.9867 | +0.0032 | 1 of 7 | −0.0028 to +0.0093 |
| Half-life: 2 years | 0.9831 | −0.0003 | 3 of 7 | −0.0075 to +0.0069 |
| Half-life: 5 years | **0.9807** | **−0.0027** | 6 of 7 | −0.0065 to +0.0010 |

Two kinds of uncertainty are reported:
- **Match level:** a paired bootstrap over all matches.
- **Season level:** because matches within a season are not independent, each season is also treated as one observation, with a t-interval, an exact sign test and a leave-one-season-out check.

**Results:**
- **No strategy beats the control by more than the uncertainty.** The differences are at most about 0.003 log loss, roughly an eighth of the gap to the bookmaker. Every match-level and season-level log-loss interval includes 0.
- **Shorter memory is not better.** Windows of 3–5 seasons and a 1-year half-life are worse on average, and they swing the most from season to season.
- **A 5-year half-life is the most consistent candidate.** It's better in 6 of 7 seasons (sign test p = 0.125), and every leave-one-season-out difference is negative. But its gain is about 0.3% of the log loss, its intervals reach 0, and it was the best of six strategies tried.
- **Recommendation:** keep the expanding, equally weighted history. Treat a 5-year half-life as a candidate to confirm on a season not used here, not as a proven improvement.

## Interactive prediction app

How the app builds a fixture's features and what it does and doesn't use. The tabs and their behaviour are described in the [README](../README.md#the-streamlit-app).

**How a future fixture gets its features.** `src/predict.py` reuses the existing feature code instead of re-implementing it:
1. It adds the fixture as one row with no score to the matches completed strictly before its date.
2. It runs `src.feature_engineering.build_features` on that table.

Because every feature is looked up from an earlier date only, the fixture never sees its own result or any match played on the same day or later. It gets exactly the 50 features, in the same order, that a played match on that date would have in `features.csv`. That includes season-to-date statistics, form carried across seasons, spells and promoted-team flags, and missing history.

`tests/test_predict.py` checks this **parity**:
- **Real data:** rebuilt historical matches equal their saved rows, including missing values. This covers random matches, each season's opening day and promoted clubs' first matches.
- **Synthetic leagues:** a synthetic league exercises same-day leakage, season transitions, promotion after a gap, a club's first ever match, future fixtures and invalid inputs.

**What the app does and doesn't use:**
- **Saved models only:** the models from `python -m src.train`, with the feature order saved alongside each one. They were trained on 1993-94 to 2025-26.
- **2026-27 results for features only:** completed 2026-27 matches feed into form and season-to-date features, but the models are never retrained on them.
- **No network calls:** the app has no external APIs.
- **Date limits:** dates outside the supported seasons, or after the end of the current season (when next season's clubs aren't known), produce a clear error message.
- **Latest data date:** the app always shows the date of the latest completed match in the data. Matches played after it aren't reflected in the predictions.
- **No experimental models:** Elo features, the Poisson goals model, the direct and recalibrated event models and the adaptive scoring rates are research results only; none feeds the app's predictions.

`tests/test_app.py` checks the app itself with Streamlit's `AppTest`:
- the three tabs and their order;
- that missing research files produce explanations, not errors, while predictions still render;
- the stale-data and distant-fixture warnings, and their thresholds;
- the training-season warning in historical demo mode;
- that the research summaries match the saved results, including the fallback when the adaptive-rate results are missing.

These are statistical estimates, not guarantees or betting recommendations.

## Goals, Over/Under and BTTS analysis

The app's **Goals & BTTS history** tab, separate from the match-outcome predictions, describes how many goals each club's own recent Serie A matches produced. `src/goal_stats.py` computes the statistics; the 1X2 models, features and predictions are unchanged.

**For each club:**
- **Goals:** scored, conceded and total (both clubs) per match.
- **Scoring:** share of matches in which the club scored, and share of clean sheets.
- **Over / Under 0.5, 1.5, 2.5 and 3.5 goals:** using the total goals of both clubs in each match.
- **BTTS Yes / No:** Yes when both clubs scored at least once.

Each percentage is shown with its count and exact denominator, e.g. `7 of 10 (70%)`.

**Historical windows** (a selector in the section):

| Window | Matches used |
|---|---|
| Last 5 / 10 / 20 matches | The club's latest matches, home and away |
| Current season to date | All of the club's matches this season so far |
| Last 5 / 10 home / away matches | Home matches for the home club, away matches for the away club |

**Rules:**
- **Time safety:** only matches completed strictly before the fixture date are used. Matches on the same day, the fixture itself and later results are never used, so historical demo mode shows exactly what was known that morning.
- **Spells, as in the features:** rolling windows run across the summer break within a club's current Serie A spell. *Current season to date* restarts every season.
- **Promoted and returning clubs** start with no history: Serie B matches are not in the data, and matches from an earlier Serie A spell (often years ago) are not used.
- **Sample sizes** are shown for both clubs. A short window uses the matches that exist and is flagged (e.g. `3 of 10 matches`). With no matches the statistics are shown as missing, never as 0%.
- **Not probabilities:** these are descriptive frequencies of past matches. They are not predictions for the fixture, have not been validated as probabilities, and the app keeps them in a separate section from the model predictions.

**Reusable code.** `team_goal_rows` reuses `order_matches` and `team_match_rows` from `src/feature_engineering.py`: one row per club per completed match, with venue, season, spell, goals for and against, and goal-line and BTTS flags. `team_sample` and `fixture_goal_stats` select a window; `goal_statistics` and `total_goals_distribution` summarise it. The Poisson goals model below reuses `team_goal_rows` for its club histories.

`tests/test_goal_stats.py` checks:
- goal lines, BTTS and averages against hand-calculated matches;
- home/away filtering, rolling windows and sample sizes;
- same-day and future-result leakage, and historical fixtures;
- missing history, promoted and returning clubs, and season boundaries;
- reproducibility;
- parity with the saved form features (`form5_gf`, `venue_form5_gf`, `season_gf`, `history_matches`, ...) on synthetic and real data;
- the app section, including that the predictions it shows are unchanged.

## Elo rating experiment

Do Elo ratings tell the models anything that the existing form, season and previous-season features don't? `src/elo.py` computes leakage-safe ratings, and `src/elo_experiment.py` runs the walk-forward backtest with and without them:

```powershell
python -m src.elo_experiment     # 3 models x 7 seasons x 2 arms, about 13 minutes
```

Results are saved to `outputs/elo/` (git-ignored). [notebooks/08_elo_rating_analysis.ipynb](../notebooks/08_elo_rating_analysis.ipynb) presents the ratings and the comparison (it loads the saved run, or runs it if missing).

**The rating update.** `E_home = 1 / (1 + 10 ** ((R_away − R_home − 60) / 400))`; then `R_home += K × (S_home − E_home)` and `R_away −= K × (S_home − E_home)`, with `S_home` = 1 for a win, 0.5 for a draw, 0 for a loss.

| Setting | Value |
|---|---|
| Initial rating (every club in 1993-94; league average) | 1500 |
| K-factor | 20 |
| Home advantage | 60 Elo points |
| Summer regression | 30% toward 1500 |

These are predetermined experimental defaults. They were fixed before the experiment and never tuned.

**Policies:**
- **No leakage:** matches are processed one calendar date at a time. Every match of a date gets its pre-match ratings from results strictly before that date, and only then are that date's results applied.
- **Promoted clubs:** no Serie B results are invented. A promoted club enters at the mean rating of the clubs that left the league that summer, so the league average stays at 1500.
- **Returning clubs:** a club coming back after relegation does not keep its old rating; it enters like any promoted club.
- **Features:** `home_elo`, `away_elo` and `elo_difference` are added as extra columns. The 50 original features, the saved production models and the app are unchanged.
- **Expected score:** `home_elo_expected` (`E_home`) is kept for analysis only. It is an expected score in which a draw counts as half a win, not the probability of a home win.

**The experiment:**
- **Control:** the existing `all` and `compact` feature sets.
- **Elo:** the same sets plus the three Elo columns.
- **Identical procedure:** both arms run the commit-10 fold procedure (past-only choice of feature set and hyperparameters, then refit) for Logistic Regression, Random Forest and XGBoost.
- **Identical fixtures:** both arms are scored on the same 2,657 fixtures.
- **Untouched:** 2025-26 and 2026-27 are excluded.
- **Reproducibility check:** the control arm reproduces the commit-10 backtest exactly.
- **Decision rule, written before the run:** recommend Elo for a model only if (1) its Bonferroni-adjusted paired bootstrap interval for the pooled log-loss difference is below 0, (2) it improves at least 5 of 7 seasons, and (3) the gain survives leaving out any one season.

| Model | Pooled log loss: control → Elo | Difference (95% interval) | Seasons better | Accuracy change |
|---|---|---|---|---|
| Logistic Regression | 0.9834 → 0.9818 | −0.0016 (−0.0042 to +0.0009) | 6 of 7 | +0.11 pts |
| Random Forest | 0.9829 → 0.9790 | −0.0039 (−0.0072 to −0.0007) | 5 of 7 | +0.08 pts |
| XGBoost | 0.9808 → 0.9794 | −0.0014 (−0.0053 to +0.0024) | 4 of 7 | −0.23 pts |

The bookmaker's pooled log loss on the same fixtures is 0.9602.

**Results:**
- **Probability quality:** log loss and Brier score are slightly lower with Elo for all three models, and every leave-one-season-out difference stays negative.
- **Accuracy and macro F1** do not improve.
- **Not confirmed after correction:** after the Bonferroni correction for three models, no interval excludes 0 (Random Forest's upper end is +0.0001). Sign tests across seasons give p ≥ 0.125.
- **No promoted-club benefit:** the gain is not concentrated in the 756 fixtures with a promoted club.
- **Why the effect is small:** the Elo gap has a rank correlation of 0.87 with the previous-season points gap, so it mostly repeats information the models already have.
- **Recommendation:** do not add Elo to the production models yet; no model meets the pre-specified rule. Keep Random Forest + Elo with frozen settings as the leading candidate, to confirm on a season this experiment has not used. Any tuning of the Elo settings must happen inside past-only nested validation.

## Poisson goals model

Can we forecast *how many goals* a match will have? `src/goals_model.py` turns two expected-goals rates per fixture into goal-market probabilities, and `src/goals_backtest.py` evaluates three ways of producing those rates with the same walk-forward scheme as the 1X2 backtest:

```powershell
python -m src.goals_backtest     # 7 seasons x 3 methods, about 1 minute
```

Results are saved to `outputs/goals/` (git-ignored). [notebooks/09_goals_prediction_analysis.ipynb](../notebooks/09_goals_prediction_analysis.ipynb) presents the theory, the backtest, calibration, uncertainty and the recommendation (figures 45-56). The app, production models, saved predictions and features are unchanged.

**Probabilities from two rates.** The home side scores `Poisson(lambda_home)` goals and the away side `Poisson(lambda_away)`, independently. Then:
- **Over/Under:** total goals are `Poisson(lambda_home + lambda_away)`, so `P(Under x.5)` is the exact Poisson CDF at `x` and `P(Over x.5) = 1 - P(Under x.5)` (lines 0.5, 1.5, 2.5 and 3.5).
- **BTTS:** `P(BTTS Yes) = (1 - exp(-lambda_home)) * (1 - exp(-lambda_away))`; `P(No) = 1 - P(Yes)`.
- **Exact scores and 1X2** (1X2 for evaluation only): from a score grid that grows until less than `1e-12` of the probability lies outside it. That truncated mass is reported, and rates so large that more than `1e-9` would be lost are refused instead of being silently cut.

**Three methods, scored on the same fixtures:**

| Method | Rates come from |
|---|---|
| League average | mean home and away goals of the training seasons |
| Team attack/defence | each club's goals scored and conceded over its last N Serie A matches (strictly earlier dates, current spell only), shrunk toward the league average, times home/away league means |
| Poisson regression | two L2-regularised `PoissonRegressor`s (home goals, away goals) on the existing engineered features; no Elo |

**Walk-forward evaluation (2018-19 to 2024-25):**
- **Past-only choices:** for evaluation season S, every setting is chosen by fitting on seasons up to S-2 and scoring Poisson deviance on S-1:
  - for each regression, the feature set (`goals` 30, `compact` 22 or `all` 50) and the penalty;
  - for the team baseline, the window and the prior.
- **Refit and score:** the chosen settings are refitted on seasons up to S-1 and scored on S. All three methods predict the same 2,660 fixtures.
- **Untouched seasons:** 2025-26 and 2026-27 are refused by the code.
- **Leakage rules:** the same as the 1X2 models; club histories and features use strictly earlier dates only.
- **Market benchmark:** market-average Over/Under 2.5 odds (`BbAv` in 2018-19, `Avg` from 2019-20) are read from the raw files already on disk, with the margin removed. They cover 2,658 of the 2,660 fixtures. They are a benchmark only, never a feature. There are no BTTS or other-line odds in the data.
- **Decision rule, written before the run:** the model is ready for the app only if:
  1. it beats the league average on the three primary scores (deviance, Over/Under 2.5 log loss, BTTS log loss), with Bonferroni-adjusted intervals below 0;
  2. it does so in at least 5 of 7 seasons;
  3. it is at least as good as the team baseline;
  4. it is calibrated: every event within 2 points of the observed share, and at most 1 of 10 reliability bins outside its 95% interval.

**Pooled results (2,660 fixtures):**

| | League average | Team attack/defence | Poisson regression | Market |
|---|---|---|---|---|
| Poisson deviance (home + away) | 2.394 | 2.170 | **2.147** | - |
| Over 2.5 log loss | 0.6945 | 0.6918 | 0.6912 | **0.6786** |
| BTTS log loss | **0.6892** | 0.6895 | 0.6900 | - |
| Mean BTTS probability (observed 55.3%) | 52.8% | 52.5% | 51.5% | - |
| Mean total goals (observed 2.77) | 2.65 | 2.80 | 2.76 | - |

**Findings:**
- **Expected goals: clear skill.** The Poisson regression is 10.3% better than the league average on deviance:
  - this holds in all 7 seasons, with an adjusted interval of −0.308 to −0.191;
  - against the team baseline it is 1.1% better and wins 6 of 7 seasons, but the adjusted interval includes 0.
- **Over/Under and BTTS: no demonstrated skill.**
  - None of the 20 probability comparisons with the baselines excludes 0, even before adjustment.
  - Over 2.5 log loss is only 0.5% better than the league average, and that gain comes mostly from the high-scoring 2019-20 and 2020-21 seasons.
  - The market's Over/Under 2.5 is better by 0.0126 (95% interval 0.0078 to 0.0175).
- **Calibration and bias:**
  - **Too extreme:** the probabilities spread too much (calibration slopes 0.46 to 0.62).
  - **Draws:** under-predicted by 3.2 points; 1-1 and 2-2 happen more often than the model expects.
  - **BTTS:** under-predicted by 3.8 points.
  - **Home wins:** over-predicted by 3.5 points; home advantage was larger in older training seasons.
  - **Goal counts:** high-scoring games and 0-0 are predicted accurately, and goals are not over-dispersed relative to the model.
- **Recommendation: not ready for the app.**
  - **Rule:** it fails all four criteria of the pre-specified rule.
  - **App:** keep the historical frequencies of the goals dashboard.
  - **Next experiments, each with its own pre-specified rule:**
    - recency weighting or a time trend for scoring level and home advantage;
    - recalibration fitted on earlier seasons;
    - Dixon-Coles or bivariate Poisson for the draw and BTTS deficit.

**Assumptions and limitations:**
- **Independence:** home and away goals are assumed independent given the rates (a small positive residual dependence remains).
- **Draws:** draws and low-scoring results other than 0-0 are under-represented.
- **Variance:** it is not exceeded here; the issue is forecasts that vary too much.
- **Missing information:** lineups, injuries and team news are not available, which is the likely reason the market is better.
- **Promoted clubs:** they start with no Serie A history, although their fixtures were not forecast worse.

`tests/test_goals_model.py` checks:
- exact Over/Under and BTTS formulas, normalisation and complementary pairs, and 1X2 against the Skellam distribution;
- zero and very large expected goals, grid truncation, and historical score probabilities;
- same-day and future-result leakage, chronological training boundaries and holdout protection;
- promoted and returning clubs, and missing history;
- identical fixtures across methods, reproducibility and row-order independence;
- pooled metrics, bootstrap intervals, calibration tables and the market-odds reader.

## Direct Over/Under and BTTS experiment

Can models trained *directly* on each yes/no event beat the Poisson-derived probabilities of commit 15? `src/event_models.py` defines the direct classifiers, the event-frequency baseline, the consistency correction and a logistic calibrator; `src/event_backtest.py` runs the walk-forward comparison:

```powershell
python -m src.event_backtest     # 7 seasons, 6 methods, about 4 minutes
```

Results are saved to `outputs/events/` (git-ignored). [notebooks/10_event_prediction_analysis.ipynb](../notebooks/10_event_prediction_analysis.ipynb) presents base rates, model choices, consistency, results, calibration, the recalibration experiment, uncertainty and the decision (figures 57-63). The app, production models, saved predictions, the Elo experiment and `outputs/goals/` are unchanged; the commit-15 Poisson forecasts are recomputed with the same code and match the saved run exactly.

**Events.** `over_0_5` ... `over_3_5` mean total goals above the line, so 3 or more goals is Over 2.5. `btts_yes` means both teams scored.

**Methods, scored on the same 2,660 fixtures:**

| Method | What it is |
|---|---|
| Event frequency | training-season share of the event, Jeffreys-smoothed `(k + 0.5) / (n + 1)` |
| League average, Team attack/defence, Poisson regression | the three commit-15 Poisson forecasts |
| Direct logistic | one regularised Logistic Regression per event (`build_logistic_regression`) on the existing features; feature set (`goals`, `compact`, `all`) and `C` chosen per event |
| Poisson regression, recalibrated | separate experiment: `sigmoid(a + b * logit(p))` on the Poisson regression's probabilities |

**Chronology:**
- **Walk-forward folds (season S):** fit on seasons up to S-2, choose on S-1 by log loss, refit on seasons up to S-1, evaluate on S. The event frequencies use the same seasons.
- **Calibration:** each season's calibrator is fitted on the Poisson regression's out-of-sample walk-forward predictions for S-3, S-2 and S-1. Each of those came from a fold that never saw the season it predicted, and S itself is never used. The base model is refitted every season, so the calibrator assumes recent errors repeat. The 3-season window was fixed in advance.
- **Excluded seasons:** 2025-26 and 2026-27 are refused.

**Probability consistency.** Over probabilities must not increase with the line:
- **Correction:** fixtures where the separately trained classifiers or calibrators break this get the closest non-increasing sequence (isotonic regression with pool-adjacent-violators). The correction is deterministic, and ordered fixtures are untouched.
- **Under:** always `1 - Over`.
- **BTTS:** predicted on its own. The direct models do not form an exact-score distribution.
- **In practice:** no fixture crossed (0 of 2,660 for both candidates), so the correction changed nothing. The direct models came within 0.35 points of crossing.

**Decision rule, written before the run.** The primary metrics are Over 2.5 and BTTS log loss, judged separately. A candidate (direct logistic or recalibrated Poisson) is ready for an event only if:
1. it beats the event frequency with a Bonferroni-adjusted interval below 0 (8 comparisons: 2 candidates × 2 events × 2 comparators);
2. it is not worse than the best of the three Poisson methods for that event;
3. it beats the event frequency in at least 5 of 7 seasons and when any one season is left out;
4. its mean is within 2 points of the observed share, its calibration slope is between 0.8 and 1.2, and at most 1 of 10 reliability bins is outside its 95% interval.

**Pooled log loss (2018-19 to 2024-25):**

| | Event frequency | League average | Team attack/defence | Poisson regression | Direct logistic | Recalibrated |
|---|---|---|---|---|---|---|
| Over 2.5 | 0.6947 | 0.6945 | 0.6918 | 0.6912 | 0.6926 | **0.6908** |
| BTTS | 0.6893 | 0.6892 | 0.6895 | 0.6900 | **0.6877** | 0.6890 |
| Over 3.5 | 0.6094 | 0.6099 | 0.6089 | 0.6090 | **0.6059** | 0.6085 |

The market-average Over 2.5 odds score 0.6786 on the same fixtures (context only).

**Findings:**
- **Over 2.5:** the best method is the recalibrated Poisson regression. It is 0.0039 better than the event frequency (95% interval −0.0092 to +0.0013) and 0.0004 better than the Poisson regression. The direct logistic is *worse* than the Poisson regression.
- **BTTS:** the best method is the direct logistic. It is 0.0017 better than the event frequency (−0.0046 to +0.0014) and 0.0016 better than the best Poisson method (league average).
- **Uncertainty:** none of the 8 decision comparisons, and none of 32 secondary ones, excludes zero even before adjustment. The candidates' gains over the event frequency rest on the high-scoring 2019-20 season: leaving it out removes them.
- **Calibration:** every method is too extreme, with calibration slopes of 0.27 to 0.61 on the primary events. Recalibration fixes the averages (BTTS goes from −3.8 to +0.4 points) but not the spread, and it was hurt in 2022-23 when scoring fell after the high-scoring seasons it learned from.
- **Decision: neither event is ready, for either candidate.** No probabilities from this experiment go into the app.
- **What it shows:** the pre-match features say very little about goal totals or BTTS beyond the base rate, and season-level drift in scoring is the largest source of error.

**Limitations:**
- **Sample size:** 2,660 fixtures cannot resolve differences of a few thousandths of a log-loss unit.
- **Noisy tuning:** settings are chosen on a single season each.
- **Calibration slopes:** pooled slopes mix fixture-level spread with season-level shifts.
- **Fixed calibration window:** it is 3 seasons and was not tuned.
- **No joint distribution:** the direct models give separate event probabilities only.
- **Missing information:** lineups and team news are not available.

`tests/test_event_models.py` checks:
- goal thresholds and BTTS;
- the smoothed event frequency;
- the monotone correction (examples, equality with isotonic regression, idempotence, Under complements);
- probability validity and the classifiers' fallback when training has only one outcome;
- the calibrator, including recovering a known distortion;
- calibration leakage: only earlier out-of-sample seasons are used, and the season being calibrated is refused;
- same-day and future-result leakage, chronological folds and holdout protection;
- identical fixtures, equality with the commit-15 forecasts and reproducibility;
- pooled metrics, the 8-comparison decision family and the rule.

## Adaptive scoring rates experiment

Commits 15 and 16 found that season-to-season drift in scoring is the largest source of error in the goals forecasts. This experiment tests whether following the most recent completed league results helps. `src/scoring_drift.py` defines the two adaptive methods, and `src/scoring_drift_backtest.py` compares them with every existing method:

```powershell
python -m src.scoring_drift_backtest     # needs outputs/goals/ and outputs/events/; about 10 seconds
```

Nothing that already exists is refitted. The six existing methods are read from `outputs/events/`, and the Poisson regression's expected goals from `outputs/goals/`. Both are checked against each other and against the training seasons, and neither folder is written to. Results go to `outputs/scoring_drift/` (git-ignored). [notebooks/11_scoring_drift_analysis.ipynb](../notebooks/11_scoring_drift_analysis.ipynb) presents the drift, the adaptive forecasts, results, calibration, drift diagnostics, uncertainty and the decision (figures 64-71). The app and production models are unchanged.

**Candidates.** Both use a window of the last **200 completed matches** and **200 pseudo-matches** of shrinkage toward history. Both settings were fixed in advance and never tuned:

| Method | What it is |
|---|---|
| Adaptive event frequency | `(events in window + 200 × static frequency) / (matches in window + 200)` for each Over line and BTTS |
| Adaptive Poisson regression | the commit-15 Poisson regression's home and away expected goals, each multiplied by `shrunk recent mean / training mean`; Over/Under and BTTS recomputed with the Poisson formulas (unadjusted when the window is empty) |

**Chronology:**
- **Model fitting:** the static frequencies, training mean goals and Poisson regression use only seasons up to S-1.
- **Online statistic:** for a fixture on date D, the window holds matches dated **strictly before D**. It may include earlier fixtures of season S, as a forecaster would have them, but never the fixture itself, other fixtures on the same date or later results.
- **Excluded seasons:** 2025-26 and 2026-27 are removed before any window is built.

**Decision rule, written before the run.** Over 2.5 and BTTS log loss are judged separately. A candidate is ready for an event only if:
1. it beats the static event frequency with a Bonferroni-adjusted interval below 0 (8 comparisons);
2. it is not worse than the strongest existing model for that event;
3. it beats the static frequency in at least 5 of 7 seasons;
4. it still beats the static frequency when any one season is left out;
5. its calibration slope is between 0.8 and 1.2, its mean is within 2 points of the observed share, and at most 1 of 10 reliability bins is outside its 95% interval.

These seasons have already shaped earlier decisions and this hypothesis, so the results are exploratory, not independent confirmation.

**Pooled log loss (2018-19 to 2024-25, same 2,660 fixtures):**

| | Static frequency | Poisson regression | Direct logistic | Recalibrated Poisson | Adaptive frequency | Adaptive Poisson |
|---|---|---|---|---|---|---|
| Over 2.5 | 0.6947 | 0.6912 | 0.6926 | 0.6908 | 0.6887 | **0.6886** |
| BTTS | 0.6893 | 0.6900 | 0.6877 | 0.6890 | **0.6863** | 0.6865 |
| Over 1.5 | 0.5380 | 0.5347 | 0.5372 | 0.5355 | 0.5340 | **0.5322** |
| Over 3.5 | 0.6094 | 0.6090 | **0.6059** | 0.6085 | 0.6066 | 0.6083 |

The market-average Over 2.5 odds score 0.6786 on the same fixtures (context only).

**Results by candidate and event:**

| Candidate | Event | vs static (Bonferroni interval) | vs strongest existing | Seasons better | Calibration slope | Ready |
|---|---|---|---|---|---|---|
| Adaptive frequency | Over 2.5 | −0.0060 (−0.0099 to −0.0023) | −0.0021 | 6 of 7 | 1.44 | No (calibration) |
| Adaptive frequency | BTTS | −0.0030 (−0.0060 to +0.0001) | −0.0014 | 4 of 7 | 1.20 | No |
| Adaptive Poisson | Over 2.5 | −0.0061 (−0.0152 to +0.0029) | −0.0021 | 3 of 7 | 0.63 | No |
| Adaptive Poisson | BTTS | −0.0028 (−0.0080 to +0.0021) | −0.0011 | 4 of 7 | 0.73 | No |

**Findings:**
- **Over 2.5:** the adaptive frequency passes four of five criteria. It is the only comparison in the decision family whose adjusted interval excludes zero, it wins 6 of 7 seasons, and it survives leaving out any season. It fails calibration: a slope of 1.44 means it is *too timid*, moving in the right direction but by too little.
- **BTTS:** the adaptive frequency is best and well calibrated, but its interval reaches +0.0001 and it wins only 4 of 7 seasons.
- **Adaptive Poisson:** it is noisier, and its gains rest on 2019-20. Still, as an exploratory finding, it beats the plain Poisson regression on BTTS (−0.0034, 6 of 7 seasons, adjusted interval below 0).
- **No candidate is shown to beat the strongest existing model.** All intervals include zero.
- **Season bias:** the adaptive methods cut the average absolute season bias, from 5.3 points (static) to 3.0 on Over 2.5 and from 3.9 to 2.3 on BTTS. Unlike the recalibration, they did not break down when scoring fell in 2022-23.
- **Drift diagnostics:**
  - Model errors persist only weakly from block to block (correlations 0.10 to 0.32).
  - The adjustment follows the errors more than it anticipates them: it caught half of the 2019-20 burst and overshot in 2021-22.
  - Most of the gain is already there in the first half of each season. It comes from using recent seasons' level, especially the higher away scoring, instead of the 1993-onward average, more than from updating within the season.
- **Decision: no candidate is ready for either event.** Nothing goes into the app, and the settings stay as fixed: the Over 2.5 slope suggests less shrinkage would fit better, but choosing that now would tune on these seasons. A fair test would apply the adaptive frequency, with these exact settings, to seasons not yet used.

`tests/test_scoring_drift.py` checks:
- that the window holds exactly the last matches before each date;
- that same-day and later scores never change a forecast, while earlier fixtures of the season do;
- that 2025-26 and later never enter the history;
- the shrinkage formulas and the division-by-zero and empty-window guards, and that invalid expected goals are refused;
- that the Poisson formulas are reused and Over probabilities stay monotone;
- that training uses only earlier seasons and every method predicts identical fixtures;
- that existing forecasts are passed through unchanged and inconsistent inputs are refused;
- reproducibility, the 8-comparison decision rule, the diagnostics, and saving and loading.

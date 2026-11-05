# Football Match Predictor

Predicting Italian Serie A match outcomes (home win / draw / away win) from historical results and team performance statistics.

The project compares a simple baseline built on historical team form against more advanced machine-learning models (Logistic Regression, Random Forest, XGBoost).

## Project structure

```
football-match-predictor/
├── data/
│   ├── raw/            # Original downloaded CSVs (not tracked in git)
│   └── processed/      # Cleaned, feature-engineered datasets (not tracked)
├── notebooks/          # Exploration, analysis and presentation of results
├── src/                # Reusable pipeline code (loading, cleaning, features, models)
├── tests/              # Automated tests (pytest)
├── outputs/
│   ├── figures/        # Charts produced by notebooks/scripts
│   ├── models/         # Saved trained models and metrics (not tracked)
│   ├── predictions/    # Saved model predictions (not tracked)
│   ├── backtest/       # Walk-forward backtest results (not tracked)
│   └── excel/          # Generated Excel workbook (not tracked)
├── requirements.txt
└── README.md
```

Python scripts in `src/` hold the pipeline logic. Notebooks import from `src/` and are used for exploration, charts and presenting findings.

## Setup

Requires Python 3.11 or newer.

```powershell
# From the DI-Bootcamp repo root, move into the project folder
cd football-match-predictor

# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Check everything is working
pytest
```

In VS Code, choose the `.venv` interpreter (Ctrl+Shift+P → "Python: Select Interpreter") so notebooks and scripts use the same environment.

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

This reads every raw file, fixes the format differences between seasons (ragged rows, two date formats, renamed odds columns, byte-order marks), validates the result, and writes `data/processed/matches.csv`, one row per match. Load it in notebooks with `src.preprocessing.load_matches()` to keep the correct column types. [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) walks through each step.

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

This writes `data/processed/features.csv`: one row per match with identifiers, the prediction targets (`result`, plus `home_goals` and `away_goals`, copied unchanged) and 50 features, 25 for the home team (`home_…`) and 25 for the away team (`away_…`). Load it with `src.feature_engineering.load_features()`. [notebooks/02_feature_analysis.ipynb](notebooks/02_feature_analysis.ipynb) explores the features.

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

[notebooks/03_baseline_model.ipynb](notebooks/03_baseline_model.ipynb) explains the process and visualises the results.

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

Each model saves its fitted pipeline, metrics and predictions under the same naming scheme as the Logistic Regression files (`outputs/models/<model>.joblib`, `outputs/predictions/<model>_test_2026-27.csv`, …). `outputs/models/model_comparison.csv` holds the side-by-side metrics. [notebooks/04_random_forest_xgboost.ipynb](notebooks/04_random_forest_xgboost.ipynb) covers the tuning, the results, how far the models agree, and permutation feature importance.

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

[notebooks/05_probability_analysis.ipynb](notebooks/05_probability_analysis.ipynb) checks how good the models' **probabilities** are, not just their top picks, and compares them with Bet365's betting odds. Nothing is retrained or tuned: the notebook reads the predictions saved by `python -m src.train`.

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

Results are saved to `outputs/backtest/` (git-ignored). [notebooks/06_walk_forward_backtesting.ipynb](notebooks/06_walk_forward_backtesting.ipynb) runs the backtest and presents it.

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

## Roadmap

- [x] Project structure and environment
- [x] Load historical Serie A data
- [x] Import and standardise Football-Data CSVs
- [x] Clean data and initial exploratory analysis
- [x] Engineer historical team performance features
- [x] Logistic Regression baseline model
- [x] Random Forest and XGBoost models
- [x] Probability calibration and bookmaker benchmark
- [x] Excel analysis and prediction dashboard
- [x] Historical walk-forward backtesting
- [ ] Final documentation and presentation

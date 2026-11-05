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
│   └── predictions/    # Saved model predictions (not tracked)
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
python -m src.train
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

## Roadmap

- [x] Project structure and environment
- [x] Load historical Serie A data
- [x] Import and standardise Football-Data CSVs
- [x] Clean data and initial exploratory analysis
- [x] Engineer historical team performance features
- [x] Logistic Regression baseline model
- [ ] Random Forest and XGBoost models
- [ ] Compare historical vs advanced models
- [ ] Excel analysis and visualisations
- [ ] Final documentation and presentation

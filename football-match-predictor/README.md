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
│   └── models/         # Saved trained models (not tracked)
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

## Roadmap

- [x] Project structure and environment
- [x] Load historical Serie A data
- [ ] Import and standardise Football-Data CSVs
- [ ] Clean data and initial exploratory analysis
- [ ] Engineer historical team performance features
- [ ] Logistic Regression baseline model
- [ ] Random Forest and XGBoost models
- [ ] Compare historical vs advanced models
- [ ] Excel analysis and visualisations
- [ ] Final documentation and presentation

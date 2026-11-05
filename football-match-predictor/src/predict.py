"""Predictions for Serie A fixtures, including fixtures that have not been played yet.

The feature calculations are **not** re-implemented here. To build the 50
pre-match features of a fixture, the fixture is appended as one placeholder
row (no score, no result) to the matches completed **strictly before** its
date, and the existing ``src.feature_engineering.build_features`` is run on
that table. Because every feature is looked up from a strictly earlier date
(``merge_asof`` with ``allow_exact_matches=False``), the placeholder:

* never sees its own (missing) result or score;
* never sees other matches played on the same day (they are not passed in);
* gets exactly the features a played match on that date would have had in
  ``features.csv``: same names, order, spells, promoted flags, season-to-date
  restarts and cross-season form, and the same missing values.

``tests/test_predict.py`` checks this parity against the saved feature table.

Usage::

    from src.predict import fixture_features, load_saved_models, predict_fixture
    row = fixture_features(matches, "Inter", "Milan", "2026-11-22")
    probabilities = predict_fixture(load_saved_models(), row)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.evaluate import OUTCOME_NAMES, OUTCOMES, predict_probabilities
from src.feature_engineering import FEATURE_COLUMNS, build_features
from src.models import MODEL_SPECS
from src.train import feature_matrix, season_name

FIXTURE_COLUMNS = ["season", "season_start", "date", "home_team", "away_team"]
# The latest season in the data is supported up to 30 June of its second year;
# later dates would belong to a season whose clubs are not known yet.
SEASON_END = (6, 30)
PROBABILITY_TOLERANCE = 1e-6


class FixtureError(ValueError):
    """A fixture that cannot be predicted (bad teams, unsupported date, ...); the message is user-facing."""


class ModelFilesMissing(FileNotFoundError):
    """Saved models are not available; the message explains how to create them."""


# ====================================================================== seasons and teams


def completed_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Matches with a known result, in date order."""
    played = matches[matches["result"].notna()]
    return played.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)


def latest_completed_date(matches: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(completed_matches(matches)["date"].max())


@dataclass(frozen=True)
class SeasonWindow:
    """The dates for which fixtures of one season can be built, and its clubs."""

    season_start: int
    first_date: pd.Timestamp  # first match of the season in the data
    last_date: pd.Timestamp  # last supported date (last match, or 30 June for the latest season)
    teams: tuple[str, ...]
    complete: bool  # False for the latest season, which may still be in progress

    @property
    def label(self) -> str:
        return season_name(self.season_start)


def season_windows(matches: pd.DataFrame) -> dict[int, SeasonWindow]:
    """One window per season in the data, keyed by the season's starting year."""
    played = completed_matches(matches)
    latest = int(played["season_start"].max())
    windows = {}
    for season, rows in played.groupby("season_start"):
        season = int(season)
        teams = tuple(sorted(pd.unique(rows[["home_team", "away_team"]].to_numpy().ravel())))
        last = rows["date"].max()
        if season == latest:
            last = pd.Timestamp(year=season + 1, month=SEASON_END[0], day=SEASON_END[1])
        windows[season] = SeasonWindow(season, rows["date"].min(), last, teams, complete=season != latest)
    return windows


def current_season(matches: pd.DataFrame) -> SeasonWindow:
    """The latest season in the data (2026-27 at the time of writing)."""
    windows = season_windows(matches)
    return windows[max(windows)]


def resolve_season(matches: pd.DataFrame, date) -> SeasonWindow:
    """The season a fixture on ``date`` belongs to; raises ``FixtureError`` outside every season."""
    date = _as_date(date)
    windows = season_windows(matches)
    for window in windows.values():
        if window.first_date <= date <= window.last_date:
            return window
    first, latest = windows[min(windows)], windows[max(windows)]
    if date < first.first_date:
        raise FixtureError(f"{date:%d %b %Y} is before the first match in the data ({first.first_date:%d %b %Y}).")
    if date > latest.last_date:
        raise FixtureError(
            f"{date:%d %b %Y} is after the {latest.label} season (supported until {latest.last_date:%d %b %Y}). "
            "The clubs of later seasons are not known yet."
        )
    raise FixtureError(f"{date:%d %b %Y} falls between two seasons; no Serie A fixtures are played then.")


def _as_date(date) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(date)
    except (TypeError, ValueError) as error:
        raise FixtureError(f"{date!r} is not a valid date.") from error
    if pd.isna(stamp):
        raise FixtureError("A match date is required.")
    return stamp.normalize()


# ====================================================================== fixture features


def fixture_features(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    date,
    restrict_history: bool = True,
) -> pd.DataFrame:
    """The 50 pre-match features of one fixture, as one row (plus identifier columns).

    Only matches completed strictly before ``date`` are used: matches on the
    same day or later, and the fixture's own result (if it was played), are
    never seen. The columns are ``FIXTURE_COLUMNS + FEATURE_COLUMNS`` in the
    same order and with the same dtypes as ``features.csv``.

    ``restrict_history`` (default) is a speed-up with identical results; see
    ``_relevant_history``. ``restrict_history=False`` passes the full history.
    Raises ``FixtureError`` with a readable message for invalid input.
    """
    home_team, away_team = _check_teams(home_team, away_team)
    date = _as_date(date)
    window = resolve_season(matches, date)
    missing = [team for team in (home_team, away_team) if team not in window.teams]
    if missing:
        raise FixtureError(
            f"{' and '.join(missing)} did not play Serie A in {window.label} according to the data."
        )

    played = completed_matches(matches)
    # Strictly before the fixture date. On the very first matchday in the data this is
    # empty, and every history feature is missing, exactly as in features.csv.
    history = played[played["date"] < date]
    if restrict_history and not plays_twice_on_one_day(history, [home_team, away_team]):
        history = _relevant_history(history, home_team, away_team, int(played["season_start"].min()))

    placeholder = {column: pd.NA for column in matches.columns}
    placeholder |= {
        "season": season_name(window.season_start),
        "season_start": window.season_start,
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
    }
    placeholder = pd.DataFrame([placeholder]).astype(matches.dtypes.to_dict())
    features = build_features(pd.concat([history, placeholder], ignore_index=True))

    row = features[
        (features["date"] == date) & (features["home_team"] == home_team) & (features["away_team"] == away_team)
    ]
    if len(row) != 1:  # pragma: no cover - guards against future changes in build_features
        raise RuntimeError(f"Expected one fixture row, found {len(row)}")
    return row[FIXTURE_COLUMNS + FEATURE_COLUMNS].reset_index(drop=True)


OTHER_CLUB = "__other_club__"  # stands in for every club other than the two in the fixture


def _relevant_history(history: pd.DataFrame, home_team: str, away_team: str, first_season: int) -> pd.DataFrame:
    """Only the history that can change this fixture's features, with other clubs merged into one.

    A team's features depend only on its own matches: its goals for and
    against, venue, dates and seasons. The opponent's identity is never used.
    So the fixture's features are unchanged if every club other than the two
    teams is given one shared name, and if matches without either team are
    dropped. The exception is the first season in the data: it decides when
    the ``promoted`` flag is unknown, so its matches are kept.
    ``build_features`` then works on a handful of team groups instead of every
    club in Serie A history, which is much faster.

    One exception, which ``fixture_features`` guards against: if a team played
    more than one match on the same date, ``build_features`` orders those
    matches by the home team's name, and that order decides which of them fall
    inside a 5- or 10-match form window. Renaming opponents could change it,
    so the full history is used instead. This never happens in real league
    data. ``tests/test_predict.py`` checks both paths against the full history.
    """
    teams = [home_team, away_team]
    keep = history["home_team"].isin(teams) | history["away_team"].isin(teams) | (
        history["season_start"] == first_season)
    relevant = history[keep].copy()
    for column in ("home_team", "away_team"):
        relevant[column] = relevant[column].where(relevant[column].isin(teams), OTHER_CLUB)
    return relevant


def plays_twice_on_one_day(history: pd.DataFrame, teams) -> bool:
    """True if any of ``teams`` has more than one match on the same date in ``history``."""
    for team in teams:
        dates = history.loc[(history["home_team"] == team) | (history["away_team"] == team), "date"]
        if dates.duplicated().any():
            return True
    return False


def _check_teams(home_team, away_team) -> tuple[str, str]:
    if not isinstance(home_team, str) or not home_team.strip():
        raise FixtureError("Choose a home team.")
    if not isinstance(away_team, str) or not away_team.strip():
        raise FixtureError("Choose an away team.")
    if home_team == away_team:
        raise FixtureError("The home and away teams must be different.")
    return home_team, away_team


def played_fixture(matches: pd.DataFrame, home_team: str, away_team: str, date) -> pd.Series | None:
    """The completed match with these teams on this date, if there is one (for demonstration mode)."""
    played = completed_matches(matches)
    rows = played[(played["date"] == _as_date(date)) & (played["home_team"] == home_team)
                  & (played["away_team"] == away_team)]
    return None if rows.empty else rows.iloc[0]


def history_summary(matches: pd.DataFrame, date) -> dict:
    """How much data a fixture on ``date`` is based on."""
    played = completed_matches(matches)
    before = played[played["date"] < _as_date(date)]
    return {"matches": len(before), "last_date": before["date"].max() if len(before) else pd.NaT}


# ====================================================================== saved models


@dataclass
class SavedModel:
    name: str  # e.g. "random_forest"
    label: str  # e.g. "Random Forest"
    pipeline: object
    feature_columns: list[str]
    metadata: dict  # contents of <name>_metrics.json (selected settings, seasons, scores)

    @property
    def training_seasons(self) -> tuple[int, int] | None:
        seasons = self.metadata.get("seasons", {}).get("final_train")
        return tuple(seasons) if seasons else None


def load_saved_model(name: str, models_dir: Path = MODELS_DIR) -> SavedModel:
    """Load a model saved by ``python -m src.train`` (never retrains)."""
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_SPECS)}")
    path = Path(models_dir) / f"{name}.joblib"
    if not path.exists():
        raise ModelFilesMissing(
            f"{path.name} was not found in {models_dir}. Generate the models locally with "
            "`python -m src.preprocessing`, `python -m src.feature_engineering` and `python -m src.train` "
            "(see the README)."
        )
    saved = joblib.load(path)
    columns = list(saved["feature_columns"])
    unknown = set(columns) - set(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"{path.name} expects unknown feature columns: {sorted(unknown)}")
    metrics_path = Path(models_dir) / f"{name}_metrics.json"
    metadata = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return SavedModel(name, MODEL_SPECS[name].label, saved["model"], columns, metadata)


def load_saved_models(names=None, models_dir: Path = MODELS_DIR) -> dict[str, SavedModel]:
    """All saved models, keyed by display label, in the usual order (LR, RF, XGBoost)."""
    names = list(names or MODEL_SPECS)
    loaded = [load_saved_model(name, models_dir) for name in names]
    return {model.label: model for model in loaded}


def predict_fixture(models: dict[str, SavedModel], features_row: pd.DataFrame) -> pd.DataFrame:
    """Probabilities H, D, A from every model (one row per model), validated."""
    if len(features_row) != 1:
        raise ValueError("Pass exactly one fixture row")
    rows = {}
    for label, model in models.items():
        X = feature_matrix(features_row, model.feature_columns)  # the model's own column order
        probs = predict_probabilities(model.pipeline, X).iloc[0]
        check_probabilities(probs, label)
        rows[label] = probs
    return pd.DataFrame(rows).T[OUTCOMES]


def check_probabilities(probs: pd.Series, label: str = "model") -> None:
    values = probs[OUTCOMES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"{label} returned invalid probabilities: {values}")
    if abs(values.sum() - 1) > PROBABILITY_TOLERANCE:
        raise ValueError(f"{label} probabilities sum to {values.sum():.6f}, not 1")


# ====================================================================== presentation helpers


def fair_odds(probability: float) -> float:
    """Decimal odds with no margin: 1 / probability (infinite for a probability of 0)."""
    probability = float(probability)
    if probability < 0 or probability > 1 or math.isnan(probability):
        raise ValueError(f"Not a probability: {probability}")
    return math.inf if probability == 0 else 1.0 / probability


def most_likely(probs: pd.Series) -> str:
    """The outcome with the highest probability (H, D or A; ties go to the first in that order)."""
    return str(probs[OUTCOMES].astype(float).idxmax())


def outcome_table(probs: pd.Series) -> pd.DataFrame:
    """Outcome / probability / fair odds / most likely, for one model."""
    best = most_likely(probs)
    return pd.DataFrame({
        "Outcome": [OUTCOME_NAMES[o] for o in OUTCOMES],
        "Probability": [float(probs[o]) for o in OUTCOMES],
        "Fair odds": [fair_odds(probs[o]) for o in OUTCOMES],
        "Most likely": [o == best for o in OUTCOMES],
    })


def simple_average(probabilities: pd.DataFrame) -> pd.Series:
    """Unweighted mean of the models' probabilities. Not a validated or separately trained model."""
    return probabilities[OUTCOMES].mean(axis=0)

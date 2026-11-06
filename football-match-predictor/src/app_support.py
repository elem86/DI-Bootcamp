"""Helpers for the Streamlit app: data freshness and research summaries from saved results.

Nothing here trains a model or reruns an experiment. The research summaries
only read result files that the experiment scripts already wrote to
``outputs/``. Those files are git-ignored, so a fresh clone may not have them:
every loader then raises ``ResearchFileMissing``, which names the command that
creates the file, and the app shows that as an explanation instead of failing.

The decisions in ``DECISIONS`` are the documented outcomes of the
pre-specified rules in the experiment modules and notebooks 06-11. They are
fixed text; the numbers shown next to them are recomputed from the saved files.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_DIR

RESEARCH_ROOT = OUTPUTS_DIR  # the app passes this to every loader; tests point it elsewhere
STALE_AFTER_DAYS = 10  # more than a week without new results: at least one round is probably missing

# ====================================================================== data freshness


def today() -> dt.date:
    """Today's date (a function so that tests can replace it)."""
    return dt.date.today()


@dataclass(frozen=True)
class Freshness:
    """How current the match data is, for a fixture on ``fixture_date`` viewed on ``today``."""

    latest_date: dt.date
    today: dt.date
    fixture_date: dt.date
    data_age_days: int  # days from the latest completed match to today
    fixture_gap_days: int  # days from the latest completed match to the fixture
    stale_after_days: int

    @property
    def data_is_stale(self) -> bool:
        return self.data_age_days > self.stale_after_days

    @property
    def fixture_is_far_ahead(self) -> bool:
        return self.fixture_gap_days > self.stale_after_days

    def warnings(self) -> list[str]:
        """Plain-language warnings for a future fixture (empty when the data is current)."""
        messages = []
        if self.data_is_stale:
            messages.append(
                f"The latest completed match in the data is from {self.latest_date:%d %b %Y}, {self.data_age_days} "
                f"days ago. Results played since then are missing, so recent form and this season's records may "
                f"be out of date. Refresh the data (see the README) for current predictions.")
        if self.fixture_is_far_ahead:
            messages.append(
                f"This fixture is {self.fixture_gap_days} days after the latest match in the data. Any matches the "
                f"clubs play before it cannot be reflected in the prediction.")
        return messages


def data_freshness(latest_date, fixture_date, today_date=None, stale_after_days: int = STALE_AFTER_DAYS) -> Freshness:
    latest = pd.Timestamp(latest_date).date()
    fixture = pd.Timestamp(fixture_date).date()
    now = pd.Timestamp(today_date).date() if today_date is not None else today()
    return Freshness(latest, now, fixture, (now - latest).days, (fixture - latest).days, int(stale_after_days))


# ====================================================================== research artifacts


class ResearchFileMissing(FileNotFoundError):
    """A saved experiment result is not available (for example in a fresh clone)."""

    def __init__(self, path: Path, command: str, what: str):
        self.path, self.command, self.what = Path(path), command, what
        super().__init__(f"{what} not found ({path}). Create it with `{command}`.")


ARTIFACTS = {
    "backtest": ("backtest/predictions.csv", "python -m src.backtest", "The 1X2 walk-forward backtest"),
    "elo": ("elo/predictions.csv", "python -m src.elo_experiment", "The Elo experiment"),
    "goals": ("goals/predictions.csv", "python -m src.goals_backtest", "The Poisson goals experiment"),
    "events": ("events/predictions.csv", "python -m src.event_backtest", "The Over/Under and BTTS experiment"),
    "scoring_drift": ("scoring_drift/predictions.csv", "python -m src.scoring_drift_backtest",
                      "The adaptive scoring-rate experiment"),
    "holdout": ("models/model_comparison.csv", "python -m src.train", "The saved models' evaluation table"),
}


def artifact_path(root, name: str) -> Path:
    """The file of artifact ``name`` under ``root``; raises ``ResearchFileMissing`` if it does not exist."""
    relative, command, what = ARTIFACTS[name]
    path = Path(root) / relative
    if not path.exists():
        raise ResearchFileMissing(path, command, what)
    return path


def available_artifacts(root) -> dict[str, bool]:
    return {name: (Path(root) / relative).exists() for name, (relative, _, _) in ARTIFACTS.items()}


def _read(root, name: str, **kwargs) -> pd.DataFrame:
    frame = pd.read_csv(artifact_path(root, name), **kwargs)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    for column in ("forecast", "method", "home_team", "away_team", "actual"):
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    return frame


def _seasons_better(per_season: pd.DataFrame, a: str, b: str) -> int:
    return int((per_season[a] < per_season[b]).sum())


def outcome_model_summary(root=RESEARCH_ROOT) -> pd.DataFrame:
    """1X2 walk-forward results (2018-19 to 2024-25) on the fixtures every forecast covers.

    One row per forecast: matches, log loss, Brier score, accuracy, and the
    number of seasons in which it had the lowest log loss.
    """
    from src.backtest import pooled_metrics, season_ranks

    predictions = _read(root, "backtest")
    pooled = pooled_metrics(predictions)
    ranks = season_ranks(predictions)
    table = pooled[["matches", "log_loss", "brier", "accuracy", "seasons"]].copy()
    table["seasons_best"] = (ranks == 1).sum().reindex(table.index).fillna(0).astype(int)
    return table.sort_values("log_loss")


def elo_summary(root=RESEARCH_ROOT) -> pd.DataFrame:
    """Each 1X2 model with and without the Elo features: pooled log loss and seasons improved."""
    from src.backtest import MODELS, pooled_metrics, season_metrics
    from src.elo_experiment import elo_label

    predictions = _read(root, "elo")
    pooled = pooled_metrics(predictions)["log_loss"]
    per_season = season_metrics(predictions)["log_loss"].unstack("forecast")
    rows = []
    for model in MODELS:
        with_elo = elo_label(model)
        if model not in pooled.index or with_elo not in pooled.index:
            continue
        rows.append({"model": model, "log_loss": pooled[model], "log_loss_with_elo": pooled[with_elo],
                     "difference": pooled[with_elo] - pooled[model],
                     "seasons_improved": _seasons_better(per_season, with_elo, model),
                     "seasons": int(per_season[model].notna().sum())})
    return pd.DataFrame(rows).set_index("model")


def expected_goals_summary(root=RESEARCH_ROOT) -> pd.DataFrame:
    """Commit-15 expected-goals methods: pooled Poisson deviance and mean absolute error per side."""
    from src.goals_backtest import pooled_metrics

    predictions = _read(root, "goals")
    pooled = pooled_metrics(predictions)
    return pooled[["matches", "deviance", "mae_home", "mae_away"]].sort_values("deviance")


EVENT_METRICS = {"log_loss_over_2_5": "Over 2.5", "log_loss_btts_yes": "BTTS"}


def event_summary(root=RESEARCH_ROOT) -> tuple[pd.DataFrame, str]:
    """Over 2.5 and BTTS log loss of every goals-market method, with seasons better than the static frequency.

    Uses the adaptive-rate results (all eight methods) when they exist, and
    otherwise the commit-16 results (six methods). Returns the table and the
    name of the artifact it came from.
    """
    from src.event_backtest import FREQUENCY, pooled_metrics, season_metrics

    try:
        predictions, source = _read(root, "scoring_drift"), "scoring_drift"
    except ResearchFileMissing:
        predictions, source = _read(root, "events"), "events"
    pooled = pooled_metrics(predictions)
    per_season = season_metrics(predictions)
    table = pooled[list(EVENT_METRICS)].rename(columns=lambda c: f"{EVENT_METRICS[c]} log loss")
    for metric, label in EVENT_METRICS.items():
        seasons = per_season[metric].unstack("method")
        table[f"{label}: seasons better than static"] = [
            _seasons_better(seasons, method, FREQUENCY) if method != FREQUENCY else pd.NA for method in table.index]
    table["seasons"] = pooled["seasons"].astype(int)
    return table, source


def holdout_summary(root=RESEARCH_ROOT) -> pd.DataFrame:
    """The saved models' scores on the 2025-26 validation season and the 2026-27 matches played so far."""
    table = _read(root, "holdout")
    table["matches"] = table["matches"].astype(int)
    return table[["split", "model", "matches", "log_loss", "brier", "accuracy"]]


# ====================================================================== documented decisions

DECISIONS = pd.DataFrame([
    {"Research question": "Which 1X2 model?",
     "Evidence": "Walk-forward backtest, 2018-19 to 2024-25 (notebooks 06-07)",
     "Outcome": "The three models score within 0.003 log loss of each other; none is clearly better, so all three "
                "are shown side by side. The bookmaker's implied probabilities are better than all of them.",
     "In the app": "Yes: the three saved models"},
    {"Research question": "Add Elo ratings to the 1X2 models?",
     "Evidence": "Elo experiment, pre-specified rule (notebook 08)",
     "Outcome": "Small improvements in pooled log loss, but no model's adjusted interval excluded zero, so Elo "
                "did not show robust extra skill.",
     "In the app": "No"},
    {"Research question": "Predict goals with a Poisson model?",
     "Evidence": "Poisson goals backtest (notebook 09)",
     "Outcome": "Better expected goals than the league average, but its Over/Under and BTTS probabilities failed "
                "the readiness rule (no robust gain over simple frequencies, poorly calibrated).",
     "In the app": "No (descriptive history only)"},
    {"Research question": "Predict Over/Under and BTTS directly, or recalibrate the Poisson probabilities?",
     "Evidence": "Event experiment (notebook 10)",
     "Outcome": "Neither candidate met the readiness rule for Over 2.5 or BTTS.",
     "In the app": "No"},
    {"Research question": "Follow recent league scoring (adaptive rates)?",
     "Evidence": "Adaptive scoring-rate experiment (notebook 11)",
     "Outcome": "Best pooled scores so far and a clear gain over the static frequency for Over 2.5, but no "
                "candidate met every criterion (the Over 2.5 forecasts were too timid). Promising, not validated; "
                "it would need confirmation on seasons not yet used.",
     "In the app": "No"},
])

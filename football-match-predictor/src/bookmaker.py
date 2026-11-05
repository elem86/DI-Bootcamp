"""Bookmaker benchmark: Bet365 1X2 odds turned into outcome probabilities.

The odds are used **only as a benchmark** for the models' probabilities.
They are never model features: ``check_feature_columns`` in ``src/train.py``
only accepts the engineered pre-match features.

From odds to probabilities
--------------------------
Decimal odds ``o`` pay ``o`` times the stake, so the *implied* probability of
an outcome is ``1 / o``. For one match the three implied probabilities add up
to a little more than 1, e.g. odds 2.00 / 3.40 / 4.00 give
``0.500 + 0.294 + 0.250 = 1.044``. That total is the **overround**; the 4.4 %
above 1 is the bookmaker's margin. Dividing each implied probability by the
total removes the margin ("basic normalisation"):
``0.479 / 0.282 / 0.239``, which sums to 1.

Basic normalisation spreads the margin in proportion to each probability.
Bookmakers tend to put relatively more margin on long shots (the
favourite-longshot bias), so more elaborate methods exist (e.g. Shin's
method); for a benchmark, the simple method is standard and transparent.

Football-Data's ``B365H/D/A`` are *pre-closing* odds, collected a day or two
before kick-off, so they already include news the models do not see
(injuries, suspensions, expected line-ups, market information).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluate import OUTCOMES, probabilities_from_results

BOOKMAKER_LABEL = "Bookmaker (Bet365)"
# Outcome -> processed odds column (see ODDS_COLUMNS in src/preprocessing.py)
ODDS_COLUMNS = {"H": "b365_home", "D": "b365_draw", "A": "b365_away"}
# A fixture is identified by its date and the two teams.
FIXTURE_KEYS = ["date", "home_team", "away_team"]


def _odds(matches: pd.DataFrame) -> pd.DataFrame:
    """The three decimal odds as float columns ``H``, ``D``, ``A``."""
    odds = matches[[ODDS_COLUMNS[o] for o in OUTCOMES]].astype("float64")
    odds.columns = OUTCOMES
    return odds


def complete_odds(matches: pd.DataFrame) -> pd.Series:
    """True where all three odds are present and valid (decimal odds above 1)."""
    odds = _odds(matches)
    return (odds.notna() & (odds > 1.0)).all(axis=1)


def implied_probabilities(matches: pd.DataFrame) -> pd.DataFrame:
    """Raw ``1 / odds`` per outcome (still including the margin); NaN if odds are incomplete."""
    implied = 1.0 / _odds(matches)
    return implied.where(complete_odds(matches), np.nan)


def overround(matches: pd.DataFrame) -> pd.Series:
    """Sum of the three implied probabilities (1.05 = 5 % bookmaker margin); NaN if incomplete."""
    return implied_probabilities(matches).sum(axis=1, min_count=len(OUTCOMES))


def bookmaker_probabilities(matches: pd.DataFrame) -> pd.DataFrame:
    """Margin-free probabilities ``H``, ``D``, ``A`` (each row sums to 1); NaN if odds are incomplete."""
    implied = implied_probabilities(matches)
    return implied.div(implied.sum(axis=1, min_count=len(OUTCOMES)), axis=0)


def odds_coverage(matches: pd.DataFrame, by: str = "season") -> pd.DataFrame:
    """Per season: completed matches, matches with complete odds, coverage and overround."""
    played = matches[matches["result"].notna()]
    frame = pd.DataFrame(
        {
            by: played[by],
            "has_odds": complete_odds(played),
            "overround": overround(played),
        }
    )
    table = frame.groupby(by, sort=False).agg(
        matches=("has_odds", "size"),
        with_odds=("has_odds", "sum"),
        mean_overround=("overround", "mean"),
        min_overround=("overround", "min"),
        max_overround=("overround", "max"),
    )
    table.insert(2, "coverage", table["with_odds"] / table["matches"])
    return table


def fixture_odds(predictions: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Look up the bookmaker data for every predicted fixture.

    Fixtures are matched on date, home team and away team. Raises if a
    prediction has no fixture, matches more than one, or if the actual result
    stored with the prediction differs from the result in ``matches`` (which
    would mean the odds belong to a different game). Returns the bookmaker
    probabilities ``H``/``D``/``A`` plus ``overround`` and ``has_odds``, in the
    row order of ``predictions``.
    """
    if predictions.duplicated(FIXTURE_KEYS).any():
        raise ValueError("Predictions contain the same fixture more than once")
    # Same date resolution on both sides, so equal dates always match
    lookup = matches[FIXTURE_KEYS + ["result", *ODDS_COLUMNS.values()]].astype({"date": "datetime64[ns]"})
    keys = predictions[FIXTURE_KEYS + ["actual"]].astype({"date": "datetime64[ns]"})
    merged = keys.merge(
        lookup, on=FIXTURE_KEYS, how="left", validate="one_to_one", indicator=True
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} predicted fixtures were not found in the match data")
    mismatched = merged["actual"].astype(str) != merged["result"].astype(str)
    if mismatched.any():
        raise ValueError(f"{int(mismatched.sum())} fixtures have a different result in the match data")

    result = bookmaker_probabilities(merged)
    result["overround"] = overround(merged)
    result["has_odds"] = complete_odds(merged)
    result.index = predictions.index
    return result


@dataclass
class Benchmark:
    """Model and bookmaker probabilities on exactly the same fixtures.

    ``fixtures`` holds date, teams, the actual result and the overround of
    every fixture with complete odds; ``probabilities`` maps each forecast name
    (the models and ``BOOKMAKER_LABEL``) to ``H``/``D``/``A`` probabilities in
    the same row order. ``excluded`` counts fixtures dropped for missing odds.
    """

    fixtures: pd.DataFrame
    probabilities: dict[str, pd.DataFrame]
    excluded: int = 0
    total: int = field(default=0)

    @property
    def actual(self) -> pd.Series:
        return self.fixtures["actual"]

    @property
    def coverage(self) -> float:
        return len(self.fixtures) / self.total if self.total else float("nan")


def build_benchmark(predictions: dict[str, pd.DataFrame], matches: pd.DataFrame) -> Benchmark:
    """Align saved model predictions with the bookmaker on the fixtures that have complete odds.

    ``predictions`` maps a model name to its saved results table (as written
    by ``src.train.save_artifacts``). All models must cover the same fixtures
    with the same results; they are put in one common order (date, home team).
    """
    if not predictions:
        raise ValueError("Need at least one model's predictions")
    columns = FIXTURE_KEYS + ["actual"]
    ordered = {
        name: frame.sort_values(FIXTURE_KEYS, kind="stable").reset_index(drop=True)
        for name, frame in predictions.items()
    }
    reference_name, reference = next(iter(ordered.items()))
    for name, frame in ordered.items():
        if not frame[columns].equals(reference[columns]):
            raise ValueError(f"{name!r} and {reference_name!r} were not evaluated on the same fixtures")

    bookmaker = fixture_odds(reference, matches)
    keep = bookmaker["has_odds"].to_numpy()

    fixtures = reference.loc[keep, columns].copy()
    fixtures["overround"] = bookmaker.loc[keep, "overround"]
    fixtures = fixtures.reset_index(drop=True)
    probabilities = {
        name: probabilities_from_results(frame.loc[keep]).reset_index(drop=True)
        for name, frame in ordered.items()
    }
    probabilities[BOOKMAKER_LABEL] = bookmaker.loc[keep, OUTCOMES].reset_index(drop=True)
    return Benchmark(fixtures, probabilities, excluded=int((~keep).sum()), total=len(reference))

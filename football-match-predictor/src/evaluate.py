"""Evaluation helpers: outcome probabilities, metrics, calibration and uncertainty.

Outcomes are always reported in the order home win, draw, away win
(``H``, ``D``, ``A``), whatever order a fitted model stores its classes in.

Two different questions are measured:

* **Prediction accuracy** - is the *most likely* outcome right?
  (accuracy, macro F1, confusion matrix)
* **Probability quality** - are the *probabilities themselves* good?
  (log loss, Brier score, calibration). A model can pick the same winners
  as another and still give worse probabilities, e.g. 90 % where 60 % is right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from src.config import RANDOM_STATE

OUTCOMES = ["H", "D", "A"]
OUTCOME_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}
PROBABILITY_COLUMNS = ["prob_home_win", "prob_draw", "prob_away_win"]

# Probabilities are clipped away from 0 before taking logs (same idea as sklearn).
_EPS = 1e-15


def predict_probabilities(model, X: pd.DataFrame) -> pd.DataFrame:
    """Predicted probabilities as columns ``H``, ``D``, ``A`` (one row per match).

    Any outcome the model never saw in training gets probability 0.
    """
    raw = model.predict_proba(X)
    probs = pd.DataFrame(raw, columns=list(model.classes_), index=X.index)
    return probs.reindex(columns=OUTCOMES, fill_value=0.0)


def predicted_outcomes(probs: pd.DataFrame) -> pd.Series:
    """The most likely outcome for each match."""
    return probs[OUTCOMES].idxmax(axis=1)


def evaluate(y_true: pd.Series, probs: pd.DataFrame) -> dict:
    """Accuracy, macro F1, log loss, confusion matrix and per-class accuracy.

    Per-class accuracy is the share of matches with that actual outcome that
    were predicted correctly (the recall of each class).
    """
    y_true = pd.Series(y_true, index=probs.index).astype(str)
    y_pred = predicted_outcomes(probs)
    matrix = confusion_matrix(y_true, y_pred, labels=OUTCOMES)
    actual_counts = matrix.sum(axis=1)
    per_class = {
        outcome: (matrix[i, i] / actual_counts[i]) if actual_counts[i] else np.nan
        for i, outcome in enumerate(OUTCOMES)
    }
    return {
        "matches": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=OUTCOMES, average="macro", zero_division=0)),
        # sklearn's log_loss expects probability columns in sorted label order (A, D, H)
        "log_loss": float(
            log_loss(y_true, probs[sorted(OUTCOMES)].to_numpy(), labels=sorted(OUTCOMES))
        ),
        "brier": brier_score(y_true, probs),
        "per_class_accuracy": per_class,
        "confusion_matrix": pd.DataFrame(
            matrix,
            index=pd.Index(OUTCOMES, name="actual"),
            columns=pd.Index(OUTCOMES, name="predicted"),
        ),
        "predicted_share": y_pred.value_counts(normalize=True).reindex(OUTCOMES, fill_value=0).to_dict(),
    }


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Side-by-side summary of several ``evaluate`` results, keyed by model name."""
    rows = {}
    for name, metrics in results.items():
        row = {
            "matches": metrics["matches"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "log_loss": metrics["log_loss"],
            "brier": metrics["brier"],
        }
        row |= {f"accuracy_{o}": metrics["per_class_accuracy"][o] for o in OUTCOMES}
        rows[name] = row
    return pd.DataFrame(rows).T


def results_table(matches: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    """One row per match: teams, actual and predicted outcome, and probabilities."""
    table = pd.DataFrame(
        {
            "date": matches["date"],
            "home_team": matches["home_team"],
            "away_team": matches["away_team"],
            "actual": matches["result"].astype(str),
            "predicted": predicted_outcomes(probs),
        },
        index=probs.index,
    )
    table[PROBABILITY_COLUMNS] = probs[OUTCOMES].to_numpy()
    table["correct"] = table["actual"] == table["predicted"]
    return table.reset_index(drop=True)


def probabilities_from_results(results: pd.DataFrame) -> pd.DataFrame:
    """Turn a saved results table (``prob_home_win`` ...) back into ``H``/``D``/``A`` columns.

    Rows are rescaled to sum to exactly 1: models that compute in float32
    (XGBoost) save rows that sum to 1 only within ~1e-7. Rows that are further
    off than rounding can explain are rejected.
    """
    probs = results[PROBABILITY_COLUMNS].astype("float64")
    probs.columns = OUTCOMES
    totals = probs.sum(axis=1)
    if ((totals - 1.0).abs() > 1e-4).any():
        raise ValueError("Saved probabilities do not sum to 1 for every match")
    return probs.div(totals, axis=0)


# ------------------------------------------------------------------ probability quality


def _aligned(y_true, probs: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Actual outcomes and H/D/A probabilities, checked and aligned row by row."""
    missing = set(OUTCOMES) - set(probs.columns)
    if missing:
        raise ValueError(f"Probabilities are missing outcome columns {sorted(missing)}")
    probs = probs[OUTCOMES].astype("float64")
    if len(y_true) != len(probs):
        raise ValueError(f"{len(y_true)} outcomes but {len(probs)} rows of probabilities")
    y_true = pd.Series(np.asarray(y_true, dtype=str), index=probs.index)
    unknown = set(y_true) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"Unknown outcomes {sorted(unknown)}; expected {OUTCOMES}")
    if probs.isna().any().any():
        raise ValueError("Probabilities contain missing values")
    return y_true, probs


def one_hot(y_true) -> pd.DataFrame:
    """1 for the outcome that happened, 0 for the other two (columns H, D, A)."""
    y_true = pd.Series(y_true).astype(str)
    return pd.DataFrame({o: (y_true == o).astype(float) for o in OUTCOMES}, index=y_true.index)


def per_match_log_loss(y_true, probs: pd.DataFrame) -> pd.Series:
    """``-log(probability given to the actual outcome)`` for every match."""
    y_true, probs = _aligned(y_true, probs)
    columns = [OUTCOMES.index(o) for o in y_true]
    actual = probs.to_numpy()[np.arange(len(probs)), columns]
    return pd.Series(-np.log(np.clip(actual, _EPS, 1.0)), index=probs.index)


def per_match_brier(y_true, probs: pd.DataFrame) -> pd.Series:
    """Squared distance between the probabilities and what happened, for every match.

    ``sum over H, D, A of (probability - (1 if it happened else 0)) ** 2``:
    0 for a certain, correct forecast; 2 for a certain, wrong one; 0.667 for 1/3 each.
    """
    y_true, probs = _aligned(y_true, probs)
    return ((probs - one_hot(y_true)) ** 2).sum(axis=1)


def brier_score(y_true, probs: pd.DataFrame) -> float:
    """Multiclass Brier score: mean of ``per_match_brier`` (lower is better, range 0-2)."""
    return float(per_match_brier(y_true, probs).mean())


def _wilson_interval(successes, trials, z: float = 1.96):
    """95 % Wilson score interval for a proportion; well-behaved for small samples."""
    successes, trials = np.asarray(successes, float), np.asarray(trials, float)
    p = successes / trials
    centre = (p + z**2 / (2 * trials)) / (1 + z**2 / trials)
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / (1 + z**2 / trials)
    return centre - half, centre + half


def reliability_table(
    y_true, probs: pd.DataFrame, outcome: str, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Predicted probability vs observed frequency of one outcome, in bins.

    ``strategy="quantile"`` puts (almost) the same number of matches in every
    bin, which suits small samples; ``"uniform"`` uses equal-width bins
    0-0.1, 0.1-0.2, ... and drops empty ones. ``ci_low``/``ci_high`` is a 95 %
    Wilson interval for the observed frequency: with ~40 matches in a bin it is
    roughly +/- 15 percentage points. For a well-calibrated forecast,
    ``observed`` is close to ``mean_predicted`` in every bin.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome {outcome!r}; expected one of {OUTCOMES}")
    if strategy not in ("quantile", "uniform"):
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    y_true, probs = _aligned(y_true, probs)
    p = probs[outcome]
    if strategy == "quantile":
        # rank first so tied probabilities still split into equal-sized bins
        bins = pd.qcut(p.rank(method="first"), q=min(n_bins, len(p)), labels=False)
    else:
        inner_edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        bins = pd.Series(np.digitize(p, inner_edges), index=p.index)

    table = (
        pd.DataFrame({"bin": bins, "p": p, "hit": (y_true == outcome).astype(float)})
        .groupby("bin")
        .agg(
            matches=("hit", "size"),
            mean_predicted=("p", "mean"),
            observed=("hit", "mean"),
            lowest=("p", "min"),
            highest=("p", "max"),
        )
        .reset_index(drop=True)
    )
    table["ci_low"], table["ci_high"] = _wilson_interval(table["observed"] * table["matches"], table["matches"])
    return table


def calibration_summary(
    y_true, probs: pd.DataFrame, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Calibration of each outcome (rows H, D, A).

    * ``mean_predicted`` vs ``observed``: on average, does the forecast expect as
      many home wins / draws / away wins as actually happen?
      ("calibration in the large"; ``difference`` = predicted - observed)
    * ``ece``: expected calibration error, the match-weighted average gap
      ``|observed - mean_predicted|`` over the reliability bins. It depends on
      the bins and is above 0 even for a perfect forecast on a small sample,
      so only compare forecasts scored on the same matches and bins.
    * ``brier``: one-vs-rest Brier score of the outcome (the three add up to
      the multiclass Brier score).
    """
    y_true, probs = _aligned(y_true, probs)
    hits = one_hot(y_true)
    rows = {}
    for outcome in OUTCOMES:
        table = reliability_table(y_true, probs, outcome, n_bins, strategy)
        gap = (table["observed"] - table["mean_predicted"]).abs()
        rows[outcome] = {
            "mean_predicted": float(probs[outcome].mean()),
            "observed": float(hits[outcome].mean()),
            "difference": float(probs[outcome].mean() - hits[outcome].mean()),
            "ece": float((gap * table["matches"]).sum() / table["matches"].sum()),
            "brier": float(((probs[outcome] - hits[outcome]) ** 2).mean()),
        }
    return pd.DataFrame(rows).T.rename_axis("outcome")


# ------------------------------------------------------------------ uncertainty

PER_MATCH_LOSSES = {"log_loss": per_match_log_loss, "brier": per_match_brier}


def paired_bootstrap(
    y_true,
    probs_a: pd.DataFrame,
    probs_b: pd.DataFrame,
    metric: str = "log_loss",
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Bootstrap confidence interval for ``metric(a) - metric(b)`` on the same matches.

    Both forecasts are scored on every match, and the per-match differences are
    resampled with replacement (the same matches for both forecasts, hence
    *paired*). A negative difference means ``a`` has the lower (better) loss.

    Limitations: matches are treated as independent (games on the same
    weekend or involving the same teams are not quite), one season says little
    about other seasons, and the interval itself is uncertain for small samples.
    An interval that includes 0 means this sample cannot tell the two apart.
    """
    if metric not in PER_MATCH_LOSSES:
        raise ValueError(f"metric must be one of {sorted(PER_MATCH_LOSSES)}")
    if not probs_a.index.equals(probs_b.index):
        raise ValueError("Both forecasts must cover the same matches in the same order")
    loss = PER_MATCH_LOSSES[metric]
    differences = (loss(y_true, probs_a) - loss(y_true, probs_b)).to_numpy()

    rng = np.random.default_rng(random_state)
    n = len(differences)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 1000):  # in chunks to limit memory
        stop = min(start + 1000, n_resamples)
        means[start:stop] = differences[rng.integers(0, n, size=(stop - start, n))].mean(axis=1)

    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return {
        "metric": metric,
        "matches": n,
        "difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "share_a_better": float((means < 0).mean()),  # share of resamples in which a wins
    }


def bootstrap_table(
    y_true,
    forecasts: dict[str, pd.DataFrame],
    pairs: list[tuple[str, str]],
    metric: str = "log_loss",
    **kwargs,
) -> pd.DataFrame:
    """``paired_bootstrap`` for several ``(a, b)`` pairs of named forecasts."""
    rows = []
    for a, b in pairs:
        result = paired_bootstrap(y_true, forecasts[a], forecasts[b], metric=metric, **kwargs)
        includes_zero = result["ci_low"] <= 0 <= result["ci_high"]
        rows.append({"a": a, "b": b} | result | {"ci_includes_zero": includes_zero})
    return pd.DataFrame(rows)

"""Evaluation helpers: outcome probabilities, metrics and the results table.

Outcomes are always reported in the order home win, draw, away win
(``H``, ``D``, ``A``), whatever order a fitted model stores its classes in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

OUTCOMES = ["H", "D", "A"]
OUTCOME_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}
PROBABILITY_COLUMNS = ["prob_home_win", "prob_draw", "prob_away_win"]


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

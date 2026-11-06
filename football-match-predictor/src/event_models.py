"""Direct probability models for Over/Under goal lines and Both Teams To Score (BTTS).

Commit 15 derived every goals probability from two Poisson expected-goals
rates. This module predicts each yes/no event *directly* instead, and adds a
past-only recalibration of the Poisson probabilities. The walk-forward
comparison lives in ``src.event_backtest``.

Events (one binary target per fixture)
--------------------------------------
* ``over_0_5`` ... ``over_3_5``: total goals greater than the line, so a match
  with three or more goals is Over 2.5 and one with two goals is not.
* ``btts_yes``: both teams scored at least once.

``event_targets`` computes them from the final score, with the same definition
as ``src.goals_backtest.event_outcomes`` and ``src.goal_stats``.

Methods
-------
* ``EventFrequency``: the share of training matches in which the event
  happened, smoothed with a Jeffreys prior, ``(k + 0.5) / (n + 1)``. The same
  probability for every fixture in a season.
* ``DirectEventClassifier``: one L2-regularised Logistic Regression per event
  (``src.models.build_logistic_regression``: median imputation with missing
  indicators, scaling, Logistic Regression) on the existing engineered
  pre-match features. The feature set (``goals``, ``compact`` or ``all``, as in
  commit 15) and ``C`` are chosen per event on an earlier season.
* ``LogisticCalibrator``: ``P_calibrated = sigmoid(a + b * logit(p))`` fitted on
  *out-of-sample* historical predictions (see ``src.event_backtest`` for the
  chronology). ``b < 1`` pulls over-confident probabilities toward the middle.

Probability consistency
-----------------------
A fixture cannot be more likely to have 3+ goals than 2+ goals, so the Over
probabilities must not increase with the line:
``P(Over 0.5) >= P(Over 1.5) >= P(Over 2.5) >= P(Over 3.5)``.
Separately trained classifiers (and separately fitted calibrators) can break
this. ``monotone_over`` repairs it deterministically: for each fixture it
returns the closest non-increasing sequence in the least-squares sense
(isotonic regression by pool-adjacent-violators with equal weights; violating
neighbours are replaced by their average). Fixtures that are already ordered
are left unchanged. Under probabilities are then defined as ``1 - Over``.

BTTS is predicted on its own. The direct models do **not** give a coherent
exact-score distribution, and nothing here pretends they do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.goal_stats import GOAL_LINES, line_key
from src.goals_model import EVENTS, GOALS_FEATURE_SETS, played
from src.models import C_GRID, build_logistic_regression
from src.train import feature_matrix

OVER_EVENTS = [line_key(line) for line in GOAL_LINES]  # over_0_5 ... over_3_5, in increasing line order
EVENT_NAMES = list(EVENTS)  # the four Over lines and btts_yes
PRIMARY_EVENTS = ["over_2_5", "btts_yes"]
JEFFREYS = 0.5
_CLIP = 1e-6


# ====================================================================== targets


def event_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """1/0 for every event from the final score (rows without a score are refused)."""
    if frame["home_goals"].isna().any() or frame["away_goals"].isna().any():
        raise ValueError("Some rows have no final score; use played() first")
    home = frame["home_goals"].to_numpy(dtype="int64")
    away = frame["away_goals"].to_numpy(dtype="int64")
    total = home + away
    table = pd.DataFrame({line_key(line): (total > line).astype("int64") for line in GOAL_LINES}, index=frame.index)
    table["btts_yes"] = ((home > 0) & (away > 0)).astype("int64")
    return table


def check_event(event: str) -> None:
    if event not in EVENT_NAMES:
        raise ValueError(f"Unknown event {event!r}; choose from {EVENT_NAMES}")


# ====================================================================== consistency


def _pava_decreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares non-increasing fit of one short sequence (pool adjacent violators)."""
    blocks = []  # [mean, size]
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            mean_b, size_b = blocks.pop()
            mean_a, size_a = blocks.pop()
            size = size_a + size_b
            blocks.append([(mean_a * size_a + mean_b * size_b) / size, size])
    return np.concatenate([np.full(size, mean) for mean, size in blocks])


def crossing_fixtures(over: pd.DataFrame, tolerance: float = 0.0) -> pd.Series:
    """True for fixtures whose Over probabilities increase somewhere along the lines."""
    values = over[OVER_EVENTS].to_numpy(dtype="float64")
    return pd.Series((np.diff(values, axis=1) > tolerance).any(axis=1), index=over.index)


def monotone_over(over: pd.DataFrame) -> pd.DataFrame:
    """Over probabilities made non-increasing in the line (see the module docstring).

    Rows that are already ordered are returned unchanged; others get the
    closest ordered sequence. Values stay inside [0, 1].
    """
    values = over[OVER_EVENTS].to_numpy(dtype="float64")
    if np.isnan(values).any():
        raise ValueError("Over probabilities contain missing values")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Over probabilities must lie in [0, 1]")
    fixed = values.copy()
    broken = np.where((np.diff(values, axis=1) > 0).any(axis=1))[0]
    for row in broken:
        fixed[row] = _pava_decreasing(values[row])
    return pd.DataFrame(fixed, columns=OVER_EVENTS, index=over.index)


def under_from_over(over: pd.DataFrame) -> pd.DataFrame:
    """``under_x_5 = 1 - over_x_5`` for every line."""
    return pd.DataFrame({line_key(line, "under"): 1.0 - over[line_key(line)] for line in GOAL_LINES}, index=over.index)


def check_valid(probabilities: pd.DataFrame) -> None:
    """Refuse missing, out-of-range or non-monotone Over probabilities, and invalid BTTS."""
    columns = [c for c in EVENT_NAMES if c in probabilities.columns]
    values = probabilities[columns].to_numpy(dtype="float64")
    if np.isnan(values).any() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be present and inside [0, 1]")
    if set(OVER_EVENTS) <= set(probabilities.columns) and crossing_fixtures(probabilities, 1e-12).any():
        raise ValueError("Over probabilities increase with the goal line for some fixtures")


# ====================================================================== baselines and classifiers


def jeffreys_rate(hits: int, trials: int) -> float:
    """``(hits + 0.5) / (trials + 1)``: a smoothed share that is never exactly 0 or 1."""
    return (hits + JEFFREYS) / (trials + 2 * JEFFREYS)


class EventFrequency:
    """Smoothed share of training matches in which each event happened."""

    label = "Event frequency"

    def fit(self, train: pd.DataFrame) -> "EventFrequency":
        train = played(train)
        if train.empty:
            raise ValueError("No training matches")
        targets = event_targets(train)
        self.rates_ = {e: jeffreys_rate(int(targets[e].sum()), len(train)) for e in EVENT_NAMES}
        self.training_matches_ = len(train)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({e: np.full(len(frame), self.rates_[e]) for e in EVENT_NAMES}, index=frame.index)


@dataclass
class EventChoice:
    """Feature set and regularisation strength of one event's classifier."""

    feature_set: str
    C: float


@dataclass
class DirectEventClassifier:
    """One Logistic Regression per event, each with its own feature set and ``C``."""

    choices: dict = field(default_factory=lambda: {e: EventChoice("goals", 0.01) for e in EVENT_NAMES})
    feature_sets: dict = field(default_factory=lambda: dict(GOALS_FEATURE_SETS))

    label = "Direct logistic"

    @property
    def params(self) -> dict:
        return {f"{e}_{k}": v for e, choice in self.choices.items() for k, v in vars(choice).items()}

    def columns(self, event: str) -> list[str]:
        return self.feature_sets[self.choices[event].feature_set]

    def fit(self, train: pd.DataFrame) -> "DirectEventClassifier":
        train = played(train)
        targets = event_targets(train)
        self.models_, self.constant_ = {}, {}
        for event, choice in self.choices.items():
            y = targets[event].to_numpy()
            if y.min() == y.max():  # only one outcome in training: fall back to the smoothed share
                self.constant_[event] = jeffreys_rate(int(y.sum()), len(y))
                continue
            model = build_logistic_regression(choice.C)
            self.models_[event] = model.fit(feature_matrix(train, self.columns(event)), y)
        return self

    def predict_event(self, frame: pd.DataFrame, event: str) -> np.ndarray:
        if event in self.constant_:
            return np.full(len(frame), self.constant_[event])
        model = self.models_[event]
        positive = list(model.classes_).index(1)
        return model.predict_proba(feature_matrix(frame, self.columns(event)))[:, positive]

    def predict(self, frame: pd.DataFrame, monotone: bool = True) -> pd.DataFrame:
        """Raw per-event probabilities; with ``monotone=True`` the Over lines are made consistent."""
        raw = pd.DataFrame({e: self.predict_event(frame, e) for e in self.choices}, index=frame.index)
        if monotone and set(OVER_EVENTS) <= set(raw.columns):
            raw[OVER_EVENTS] = monotone_over(raw)
        return raw

    def coefficients(self, event: str) -> pd.Series:
        model = self.models_[event]
        names = model[:-1].get_feature_names_out(self.columns(event))
        return pd.Series(model[-1].coef_[0], index=names)


def binary_log_loss(probability, outcome) -> np.ndarray:
    """Per-match ``-(y log p + (1 - y) log(1 - p))`` (probabilities clipped away from 0 and 1)."""
    p = np.clip(np.asarray(probability, dtype="float64"), 1e-15, 1 - 1e-15)
    y = np.asarray(outcome, dtype="float64")
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def tune_event(train: pd.DataFrame, validation: pd.DataFrame, event: str, feature_sets=None,
               c_grid=C_GRID) -> pd.DataFrame:
    """Validation log loss of every feature set x C for one event's classifier (best first)."""
    check_event(event)
    feature_sets = feature_sets or GOALS_FEATURE_SETS
    train, validation = played(train), played(validation)
    y_train = event_targets(train)[event].to_numpy()
    y_val = event_targets(validation)[event].to_numpy()
    rows = []
    for name, columns in feature_sets.items():
        X_train, X_val = feature_matrix(train, columns), feature_matrix(validation, columns)
        for C in c_grid:
            if y_train.min() == y_train.max():
                p = np.full(len(validation), jeffreys_rate(int(y_train.sum()), len(y_train)))
            else:
                model = build_logistic_regression(C).fit(X_train, y_train)
                p = model.predict_proba(X_val)[:, list(model.classes_).index(1)]
            rows.append({"event": event, "feature_set": name, "features": len(columns), "C": C,
                         "log_loss": float(binary_log_loss(p, y_val).mean())})
    return pd.DataFrame(rows).sort_values("log_loss", kind="stable").reset_index(drop=True)


# ====================================================================== recalibration


def logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype="float64"), _CLIP, 1 - _CLIP)
    return np.log(p / (1 - p))


@dataclass
class LogisticCalibrator:
    """``sigmoid(intercept + slope * logit(p))``, fitted on earlier out-of-sample predictions.

    If the calibration data hold only one outcome, or the probabilities do not
    vary, the calibrator stays the identity (intercept 0, slope 1) and
    ``fitted`` is False.
    """

    intercept: float = 0.0
    slope: float = 1.0
    fitted: bool = False
    matches: int = 0

    def fit(self, probability, outcome) -> "LogisticCalibrator":
        x = logit(probability).reshape(-1, 1)
        y = np.asarray(outcome).astype(int)
        self.matches = len(y)
        if len(y) == 0 or y.min() == y.max() or np.ptp(x) < 1e-9:
            self.intercept, self.slope, self.fitted = 0.0, 1.0, False
            return self
        model = LogisticRegression(C=np.inf, max_iter=1000).fit(x, y)
        self.intercept, self.slope, self.fitted = float(model.intercept_[0]), float(model.coef_[0, 0]), True
        return self

    def transform(self, probability) -> np.ndarray:
        z = self.intercept + self.slope * logit(probability)
        return 1.0 / (1.0 + np.exp(-z))

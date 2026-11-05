"""Model definitions and tuning grids for the match-outcome classifiers.

Every model is a scikit-learn ``Pipeline`` that takes the raw feature matrix
(with NaN for missing history) and predicts ``H`` / ``D`` / ``A``:

* **Logistic Regression**: median imputation + missing indicators,
  standardisation, L2-regularised multinomial Logistic Regression.
* **Random Forest**: median imputation + missing indicators, then a forest of
  decision trees. Trees don't need scaled inputs.
* **XGBoost**: gradient-boosted trees, which handle missing values natively
  (each split learns which way missing values should go), so no imputation.

All preprocessing lives inside the pipeline, so it is fitted on training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

# ------------------------------------------------------------------ builders


def build_logistic_regression(C: float = 1.0) -> Pipeline:
    """Median imputation (+ missing indicators) -> scaling -> Logistic Regression.

    ``add_indicator=True`` adds a 0/1 column for every feature that had missing
    values in training, so the model knows *that* history was missing (e.g. a
    promoted team without last-season stats), not just the filled-in value.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def build_random_forest(
    n_estimators: int = 400,
    max_depth: int | None = 8,
    min_samples_leaf: int = 50,
    max_features: str | float = "sqrt",
) -> Pipeline:
    """Median imputation (+ missing indicators) -> Random Forest.

    ``min_samples_leaf`` and ``max_depth`` limit how finely each tree can
    split, which keeps the predicted probabilities smooth instead of
    over-confident.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


class XGBOutcomeClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost multiclass classifier that accepts string labels (``H``, ``D``, ``A``).

    ``XGBClassifier`` only accepts labels ``0..k-1``; this wrapper encodes the
    labels on ``fit`` and exposes ``classes_`` like any scikit-learn classifier.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        max_depth: int = 3,
        min_child_weight: float = 10.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int = RANDOM_STATE,
        n_jobs: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        labels = np.asarray(y).astype(str)
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Need at least two different outcomes to train.")
        self.booster_ = XGBClassifier(
            objective="multi:softprob" if len(self.classes_) > 2 else "binary:logistic",
            eval_metric="mlogloss" if len(self.classes_) > 2 else "logloss",
            tree_method="hist",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.booster_.fit(X, encoded)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "booster_")
        return self.booster_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def feature_importances_(self):
        check_is_fitted(self, "booster_")
        return self.booster_.feature_importances_


def build_xgboost(
    n_estimators: int = 300,
    learning_rate: float = 0.03,
    max_depth: int = 3,
    min_child_weight: float = 10.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> Pipeline:
    """XGBoost on the raw features (missing values handled natively)."""
    return Pipeline(
        [
            (
                "model",
                XGBOutcomeClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                ),
            )
        ]
    )


# ------------------------------------------------------------------ registry


def grid(**options) -> list[dict]:
    """All combinations of the given parameter options, as a list of dicts."""
    keys = list(options)
    return [dict(zip(keys, values)) for values in product(*options.values())]


# Inverse regularisation strength: smaller C = stronger L2 penalty, smaller coefficients.
C_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


@dataclass(frozen=True)
class ModelSpec:
    name: str  # file-name friendly, e.g. "random_forest"
    label: str  # display name, e.g. "Random Forest"
    build: Callable[..., Pipeline]
    param_grid: list[dict]


MODEL_SPECS = {
    spec.name: spec
    for spec in [
        ModelSpec(
            "logistic_regression",
            "Logistic Regression",
            build_logistic_regression,
            grid(C=C_GRID),
        ),
        ModelSpec(
            "random_forest",
            "Random Forest",
            build_random_forest,
            # Deeper trees with larger leaves vs shallower trees: all regularised,
            # because match outcomes are noisy and unconstrained trees overfit.
            grid(max_depth=[6, 12, None], min_samples_leaf=[10, 20, 50, 100]),
        ),
        ModelSpec(
            "xgboost",
            "XGBoost",
            build_xgboost,
            # Small learning rate with shallow trees; the number of trees sets
            # how far boosting goes before it starts fitting noise.
            grid(max_depth=[2, 3, 4], n_estimators=[100, 250, 500]),
        ),
    ]
}

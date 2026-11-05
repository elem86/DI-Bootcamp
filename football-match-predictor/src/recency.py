"""Recency strategies for training: rolling windows and exponential sample weights.

Usage (from the project folder)::

    python -m src.recency            # all strategies, seasons 2018-19 to 2024-25

Results are saved to ``outputs/recency/`` (git-ignored).

Three ways to use the training history
--------------------------------------
* **Expanding, equal weights (control):** every earlier match counts the same.
  This is the commit-10 procedure.
* **Rolling window of k seasons:** only the k most recent training seasons are
  used; older matches are dropped.
* **Exponential decay with half-life h years:** every training match is kept,
  but older matches count less.

Weight formula
--------------
For a training set whose most recent match was played on ``cutoff``::

    age_i = (cutoff - date_i) in days / 365.25       # years before the cutoff
    raw_i = 0.5 ** (age_i / h)                       # = exp(-ln(2) * age_i / h)
    w_i   = raw_i / mean(raw)                        # normalised: mean weight = 1

A match played h years before the cutoff weighs half as much as the most
recent one, 2h years before a quarter, and so on. Normalising to a mean of 1
keeps the total weight equal to the number of rows, the same as the equally
weighted control, so the regularisation strength ``C`` keeps a comparable
meaning. The *effective* sample size ``(sum w)^2 / sum(w^2)`` (Kish) shows how
many equally weighted matches the weighted training set is worth.

Weights are passed to the Logistic Regression step only; the median imputer
and the scaler are fitted on the same rows without weights.

The cutoff is the training set's own last match: during inner tuning (train up
to S-2) it is the last match of S-2, and for the final fit (train up to S-1)
the last match of S-1. The same strategy is applied in both steps.

Features are unchanged: a rolling window only decides which matches are
*training examples*. Each match's pre-match features still describe its own
history, which was known before kick-off.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import (
    BACKTEST_SEASONS,
    BASELINE,
    Backtest,
    pooled_metrics,
    run_backtest,
    save_backtest,
    season_metrics,
)
from src.bookmaker import BOOKMAKER_LABEL, FIXTURE_KEYS
from src.config import OUTPUTS_DIR
from src.models import MODEL_SPECS
from src.train import TARGET_COLUMN, season_name

RECENCY_DIR = OUTPUTS_DIR / "recency"
DAYS_PER_YEAR = 365.25
LOWER_IS_BETTER = {"log_loss": True, "brier": True, "accuracy": False, "macro_f1": False}


# ====================================================================== weights and windows


def match_age_years(dates, cutoff=None) -> np.ndarray:
    """Age of every match in years before ``cutoff`` (default: the most recent date)."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    cutoff = dates.max() if cutoff is None else pd.Timestamp(cutoff)
    age = ((cutoff - dates) / pd.Timedelta(days=1)).to_numpy(dtype=float) / DAYS_PER_YEAR
    if (age < 0).any():
        raise ValueError("Training matches cannot be played after the training cutoff")
    return age


def exponential_weights(dates, half_life_years: float, cutoff=None) -> np.ndarray:
    """``0.5 ** (age / half_life)``, normalised so the mean weight is 1."""
    if not half_life_years or half_life_years <= 0:
        raise ValueError("half_life_years must be positive")
    raw = 0.5 ** (match_age_years(dates, cutoff) / half_life_years)
    return raw / raw.mean()


def effective_sample_size(weights) -> float:
    """Kish's effective sample size: (sum w)^2 / sum(w^2); equals n for equal weights."""
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.square(weights).sum())


def rolling_window(train: pd.DataFrame, window_seasons: int) -> pd.DataFrame:
    """Rows from the ``window_seasons`` most recent seasons of ``train`` (all rows if fewer exist)."""
    if not isinstance(window_seasons, (int, np.integer)) or window_seasons < 1:
        raise ValueError("window_seasons must be a positive integer")
    last = int(train["season_start"].max())
    return train[train["season_start"] > last - window_seasons]


@dataclass(frozen=True)
class RecencyStrategy:
    """How to use the training history. At most one of window / half-life is set."""

    name: str  # file-name friendly, e.g. "window_5"
    label: str  # display name
    window_seasons: int | None = None
    half_life_years: float | None = None

    def __post_init__(self):
        if self.window_seasons is not None and self.half_life_years is not None:
            raise ValueError("Use either a rolling window or a half-life, not both")
        if self.window_seasons is not None and (not isinstance(self.window_seasons, int) or self.window_seasons < 1):
            raise ValueError("window_seasons must be a positive integer")
        if self.half_life_years is not None and self.half_life_years <= 0:
            raise ValueError("half_life_years must be positive")

    @property
    def kind(self) -> str:
        if self.window_seasons is not None:
            return "rolling window"
        if self.half_life_years is not None:
            return "exponential decay"
        return "expanding"

    @property
    def parameter(self) -> str:
        if self.window_seasons is not None:
            return f"{self.window_seasons} seasons"
        if self.half_life_years is not None:
            unit = "year" if self.half_life_years == 1 else "years"
            return f"half-life {self.half_life_years:g} {unit}"
        return "all seasons, equal weights"

    def prepare(self, train: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Training rows to keep and their sample weights (``None`` = equal weights)."""
        if self.window_seasons is not None:
            return rolling_window(train, self.window_seasons), None
        if self.half_life_years is not None:
            return train, exponential_weights(train["date"], self.half_life_years)
        return train, None


CONTROL = RecencyStrategy("expanding", "Expanding (control)")
STRATEGIES = (
    CONTROL,
    RecencyStrategy("window_3", "Window: 3 seasons", window_seasons=3),
    RecencyStrategy("window_5", "Window: 5 seasons", window_seasons=5),
    RecencyStrategy("window_10", "Window: 10 seasons", window_seasons=10),
    RecencyStrategy("half_life_1", "Half-life: 1 year", half_life_years=1),
    RecencyStrategy("half_life_2", "Half-life: 2 years", half_life_years=2),
    RecencyStrategy("half_life_5", "Half-life: 5 years", half_life_years=5),
)


def training_footprint(strategy: RecencyStrategy, train: pd.DataFrame) -> dict:
    """Rows used, seasons covered and effective sample size of one training set."""
    rows, weights = strategy.prepare(train)
    w = np.ones(len(rows)) if weights is None else weights
    in_last_season = rows["season_start"].to_numpy() == rows["season_start"].max()
    return {
        "rows_used": len(rows),
        "first_season": int(rows["season_start"].min()),
        "last_season": int(rows["season_start"].max()),
        "effective_matches": effective_sample_size(w),
        "weight_of_last_season": float(w[in_last_season].sum() / w.sum()),  # share of the total weight
    }


def strategy_table(strategies, train: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy: what it does and how much of ``train`` it effectively uses."""
    played = train[train[TARGET_COLUMN].notna()]
    rows = []
    for strategy in strategies:
        footprint = training_footprint(strategy, played)
        rows.append({"strategy": strategy.label, "kind": strategy.kind, "setting": strategy.parameter} | footprint)
    return pd.DataFrame(rows).set_index("strategy")


# ====================================================================== experiment


def run_recency_backtest(
    features: pd.DataFrame,
    matches: pd.DataFrame,
    strategies=STRATEGIES,
    seasons=BACKTEST_SEASONS,
    model_name: str = "logistic_regression",
    param_grid: list[dict] | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    progress=None,
) -> Backtest:
    """Walk-forward backtest of one model under every strategy, on identical fixtures.

    Each strategy runs the full commit-10 fold procedure (past-only tuning of
    feature set and hyperparameters, then refit) with the strategy applied to
    both the tuning fits and the final fit. The baseline and bookmaker rows are
    kept once; they do not depend on the strategy.
    """
    strategies = list(strategies)
    if len({s.label for s in strategies}) != len(strategies):
        raise ValueError("Strategy labels must be unique")
    label = MODEL_SPECS[model_name].label
    param_grids = {model_name: param_grid} if param_grid is not None else None
    frames, folds, reference = [], [], None
    for strategy in strategies:
        report = (lambda message, s=strategy: progress(f"[{s.label}] {message}")) if progress else None
        backtest = run_backtest(features, matches, seasons=seasons, models=[model_name], param_grids=param_grids,
                                feature_sets=feature_sets, strategy=strategy, progress=report)
        preds = backtest.predictions
        model_rows = preds[preds["forecast"] == label].assign(forecast=strategy.label)
        keys = model_rows[["season", *FIXTURE_KEYS, "actual"]].reset_index(drop=True)
        if reference is None:
            reference = keys
            frames.append(preds[preds["forecast"].isin([BOOKMAKER_LABEL, BASELINE])])
        elif not keys.equals(reference):
            raise ValueError(f"{strategy.label} was evaluated on different fixtures")
        frames.append(model_rows)

        fold_rows = backtest.folds.assign(strategy=strategy.name, strategy_label=strategy.label)
        footprints = []
        for season in fold_rows["season"]:
            train = features[(features["season_start"] <= season - 1) & features[TARGET_COLUMN].notna()]
            footprints.append(training_footprint(strategy, train))
        folds.append(pd.concat([fold_rows.reset_index(drop=True), pd.DataFrame(footprints)], axis=1))

    predictions = pd.concat(frames, ignore_index=True)
    return Backtest(predictions=predictions, folds=pd.concat(folds, ignore_index=True))


# ====================================================================== analysis


def season_differences(predictions: pd.DataFrame, control: str = CONTROL.label, metric: str = "log_loss",
                       forecasts=None) -> pd.DataFrame:
    """Per season: strategy score minus control score (rows = seasons, columns = strategies)."""
    wide = season_metrics(predictions)[metric].unstack("forecast")
    forecasts = [f for f in (forecasts or wide.columns) if f != control]
    return wide[forecasts].sub(wide[control], axis=0)


def season_level_summary(predictions: pd.DataFrame, control: str = CONTROL.label, metric: str = "log_loss",
                         forecasts=None, confidence: float = 0.95) -> pd.DataFrame:
    """Treat each *season* as one observation of (strategy - control).

    Matches within a season share conditions, so this complements the
    match-level bootstrap with a coarser but more conservative view:

    * ``mean_difference`` with a t-interval over the seasonal differences;
    * ``seasons_better`` / ``seasons_worse`` and an exact two-sided sign-test p-value;
    * ``pooled_difference`` (every match counts once) and the range of that
      pooled difference when each season in turn is left out.

    Pooled differences are computed by ``src.backtest.pooled_metrics`` from the
    pooled per-match predictions. For log loss, Brier score and accuracy this
    equals the match-weighted mean of the seasonal differences; for macro F1 it
    does not, because F1 is not an average over matches.
    """
    lower_better = LOWER_IS_BETTER[metric]
    diffs = season_differences(predictions, control, metric, forecasts)
    pooled_all = pooled_metrics(predictions)[metric]
    # Pooled scores with each season left out in turn
    pooled_without = {
        season: pooled_metrics(predictions[predictions["season"] != season])[metric] for season in diffs.index
    } if len(diffs.index) > 1 else {}
    rows = {}
    for forecast in diffs.columns:
        d = diffs[forecast].dropna()
        n = len(d)
        better = int((d < 0).sum() if lower_better else (d > 0).sum())
        worse = int((d > 0).sum() if lower_better else (d < 0).sum())
        sd = float(d.std(ddof=1)) if n > 1 else float("nan")
        half = stats.t.ppf(0.5 + confidence / 2, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
        pooled = float(pooled_all[forecast] - pooled_all[control])
        leave_one_out = [float(pooled_without[s][forecast] - pooled_without[s][control]) for s in d.index] \
            if n > 1 else [pooled]
        rows[forecast] = {
            "seasons": n,
            "mean_difference": float(d.mean()),
            "sd": sd,
            "ci_low": float(d.mean() - half),
            "ci_high": float(d.mean() + half),
            "seasons_better": better,
            "seasons_worse": worse,
            "sign_test_p": float(stats.binomtest(better, better + worse, 0.5).pvalue) if better + worse else 1.0,
            "pooled_difference": pooled,
            "leave_one_out_min": min(leave_one_out),
            "leave_one_out_max": max(leave_one_out),
        }
    return pd.DataFrame(rows).T


def main(argv: list[str] | None = None) -> None:
    from src.feature_engineering import load_features
    from src.preprocessing import load_matches

    parser = argparse.ArgumentParser(description="Recency-weighting experiment (walk-forward backtest).")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(BACKTEST_SEASONS))
    args = parser.parse_args(argv)
    backtest = run_recency_backtest(load_features(), load_matches(), seasons=args.seasons, progress=print)
    paths = save_backtest(backtest, RECENCY_DIR)
    pd.set_option("display.width", 160)
    print("\nPooled (identical fixtures):")
    print(pooled_metrics(backtest.predictions).round(4).to_string())
    print("\nSeason-level log-loss differences vs control:")
    print(season_level_summary(backtest.predictions).round(4).to_string())
    print(f"\nSeasons: {', '.join(season_name(s) for s in sorted(args.seasons))}")
    for name, path in paths.items():
        print(f"saved {name}: {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()

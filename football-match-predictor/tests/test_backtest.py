"""Tests for the walk-forward backtest (src.backtest)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.backtest import (
    BASELINE,
    FORECASTS,
    MODELS,
    PREDICTION_COLUMNS,
    common_fixtures,
    coverage_table,
    load_backtest,
    pooled_bootstrap,
    pooled_metrics,
    run_backtest,
    run_fold,
    save_backtest,
    season_metrics,
    season_ranks,
    seasonal_bootstrap,
    summary_table,
)
from src.bookmaker import BOOKMAKER_LABEL
from src.evaluate import OUTCOMES, evaluate
from src.feature_engineering import FEATURE_COLUMNS
from src.train import TEST_SEASON
from tests.test_models import FAST_GRIDS
from tests.test_train import make_features

SEASONS = range(2000, 2008)
BACKTEST = (2004, 2005, 2006)


def make_matches(features, seed=3):
    """Match data with Bet365-style odds for every synthetic fixture; a few odds are missing."""
    rng = np.random.default_rng(seed)
    matches = features[["season_start", "date", "home_team", "away_team", "result"]].copy()
    probs = rng.dirichlet([4, 2.5, 3], size=len(matches))
    odds = np.round(1 / (probs * 1.05), 2)
    matches[["b365_home", "b365_draw", "b365_away"]] = odds
    missing = matches.index[matches["season_start"] == 2005][:7]
    matches.loc[missing, "b365_draw"] = np.nan
    return matches


@pytest.fixture(scope="module")
def data():
    features = make_features(seasons=SEASONS)
    return features, make_matches(features)


@pytest.fixture(scope="module")
def backtest(data):
    features, matches = data
    return run_backtest(features, matches, seasons=BACKTEST, param_grids=FAST_GRIDS)


# ---------------------------------------------------------------- chronology and boundaries


def test_folds_are_chronological_with_expanding_windows(backtest, data):
    features, _ = data
    folds = backtest.folds
    for model in MODELS:
        rows = folds[folds["model"] == model].reset_index(drop=True)
        assert rows["season"].tolist() == list(BACKTEST)
        assert (rows["first_train_season"] == SEASONS[0]).all()  # expanding: always from the first season
        assert (rows["last_train_season"] == rows["season"] - 1).all()
        assert (rows["tuning_season"] == rows["season"] - 1).all()
        played_before = [int((features["season_start"] < s).sum()) for s in rows["season"]]
        assert rows["train_matches"].tolist() == played_before
        assert rows["train_matches"].is_monotonic_increasing


def test_each_fold_predicts_only_its_own_season(backtest, data):
    features, _ = data
    preds = backtest.predictions
    for season in BACKTEST:
        rows = preds[(preds["season"] == season) & (preds["forecast"] == MODELS[0])]
        season_dates = features.loc[features["season_start"] == season, "date"]
        assert set(rows["date"]) == set(season_dates)
        earlier = features.loc[features["season_start"] < season, "date"]
        assert rows["date"].min() > earlier.max()  # every training match happened before


def test_later_seasons_cannot_influence_a_fold(data):
    features, _ = data
    changed = features.copy()
    later = changed["season_start"] > 2005
    changed.loc[later, "result"] = "D"
    changed.loc[later, FEATURE_COLUMNS[:10]] = 50.0

    a = run_fold(features, 2005, "random_forest", FAST_GRIDS["random_forest"])
    b = run_fold(changed, 2005, "random_forest", FAST_GRIDS["random_forest"])
    pd.testing.assert_frame_equal(a.predictions, b.predictions)


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "xgboost"])
def test_evaluation_season_results_never_reach_fitting_or_tuning(data, model_name):
    """Scrambling season S's results must not change the settings or the probabilities for S."""
    features, _ = data
    scrambled = features.copy()
    rows = scrambled["season_start"] == 2005
    scrambled.loc[rows, "result"] = np.random.default_rng(9).choice(OUTCOMES, size=int(rows.sum()))

    a = run_fold(features, 2005, model_name, FAST_GRIDS[model_name])
    b = run_fold(scrambled, 2005, model_name, FAST_GRIDS[model_name])

    assert (a.feature_set, a.params) == (b.feature_set, b.params)
    probability_columns = ["prob_home_win", "prob_draw", "prob_away_win"]
    pd.testing.assert_frame_equal(a.predictions[probability_columns], b.predictions[probability_columns])
    pd.testing.assert_frame_equal(a.baseline, b.baseline)


def test_tuning_uses_the_season_before(data):
    """The inner validation season is S-1, and the selected settings come from the predefined grid."""
    features, _ = data
    fold = run_fold(features, 2005, "logistic_regression", FAST_GRIDS["logistic_regression"])
    assert fold.tuning_season == 2004 and fold.last_train_season == 2004
    assert fold.params in FAST_GRIDS["logistic_regression"]


def test_holdout_and_too_early_seasons_are_refused(data):
    features, _ = data
    with pytest.raises(ValueError, match="holdout"):
        run_fold(features, TEST_SEASON, "logistic_regression")
    with pytest.raises(ValueError, match="two earlier seasons"):
        run_fold(features, SEASONS[0] + 1, "logistic_regression")


def test_backtest_is_reproducible(data, backtest):
    features, matches = data
    again = run_backtest(features, matches, seasons=BACKTEST, param_grids=FAST_GRIDS)
    pd.testing.assert_frame_equal(again.predictions, backtest.predictions)
    pd.testing.assert_frame_equal(again.folds, backtest.folds)


# ---------------------------------------------------------------- forecasts and fixtures


def test_every_forecast_is_present_with_valid_probabilities(backtest):
    preds = backtest.predictions
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert set(preds["forecast"]) == set(FORECASTS)
    np.testing.assert_allclose(preds[OUTCOMES].sum(axis=1), 1.0)
    for season in BACKTEST:
        counts = preds[preds["season"] == season].groupby("forecast").size()
        assert counts[MODELS].nunique() == 1 and counts[BASELINE] == counts[MODELS[0]]


def test_bookmaker_only_on_fixtures_with_complete_odds(backtest):
    preds = backtest.predictions
    book = preds[preds["forecast"] == BOOKMAKER_LABEL]
    assert book["has_odds"].all()
    coverage = coverage_table(preds)
    assert coverage.loc[2005, "fixtures"] == 120
    assert coverage.loc[2005, "with_odds"] == 113
    assert coverage.loc[2004, "coverage"] == 1.0
    assert len(book[book["season"] == 2005]) == 113


def test_metrics_use_identical_fixtures_for_every_forecast(backtest):
    seasonal = season_metrics(backtest.predictions)
    assert (seasonal.loc[2005, "matches"] == 113).all()
    assert (seasonal.loc[2004, "matches"] == 120).all()
    all_rows = season_metrics(backtest.predictions, common=False)
    assert all_rows.loc[(2005, MODELS[0]), "matches"] == 120


def test_common_fixtures_refuses_mismatched_forecasts(backtest):
    preds = backtest.predictions
    broken = preds.drop(preds.index[(preds["forecast"] == MODELS[1]) & (preds["season"] == 2004)][:1])
    with pytest.raises(ValueError, match="different fixtures"):
        common_fixtures(broken)


def test_baseline_is_the_training_outcome_frequency(backtest, data):
    features, _ = data
    preds = backtest.predictions
    for season in BACKTEST:
        expected = features.loc[features["season_start"] < season, "result"].value_counts(normalize=True)
        rows = preds[(preds["season"] == season) & (preds["forecast"] == BASELINE)]
        np.testing.assert_allclose(rows[OUTCOMES].iloc[0].to_numpy(float), expected.reindex(OUTCOMES).to_numpy())


def test_models_beat_the_baseline_on_learnable_data(backtest):
    pooled = pooled_metrics(backtest.predictions)
    for model in MODELS:
        assert pooled.loc[model, "log_loss"] < pooled.loc[BASELINE, "log_loss"]


# ---------------------------------------------------------------- aggregation


def handmade_predictions():
    """Two seasons of different sizes (4 and 2 matches) for two forecasts."""
    rows = []
    actual = {2010: ["H", "D", "A", "H"], 2011: ["A", "H"]}
    probs = {
        "Logistic Regression": {2010: [[.6, .3, .1], [.3, .4, .3], [.2, .2, .6], [.2, .3, .5]],
                                2011: [[.1, .2, .7], [.5, .3, .2]]},
        BASELINE: {2010: [[.45, .27, .28]] * 4, 2011: [[.45, .27, .28]] * 2},
    }
    for forecast, by_season in probs.items():
        for season, p in by_season.items():
            for k, (outcome, row) in enumerate(zip(actual[season], p)):
                rows.append({"season": season, "forecast": forecast, "date": pd.Timestamp(f"{season}-09-0{k + 1}"),
                             "home_team": f"T{k}", "away_team": f"U{k}", "actual": outcome,
                             "H": row[0], "D": row[1], "A": row[2], "has_odds": True})
    return pd.DataFrame(rows)[PREDICTION_COLUMNS]


def test_pooled_metrics_weight_every_match_equally():
    preds = handmade_predictions()
    seasonal = season_metrics(preds)
    pooled = pooled_metrics(preds)
    model = "Logistic Regression"

    s10, s11 = seasonal.loc[(2010, model)], seasonal.loc[(2011, model)]
    weighted = (4 * s10["log_loss"] + 2 * s11["log_loss"]) / 6
    assert pooled.loc[model, "log_loss"] == pytest.approx(weighted)
    assert pooled.loc[model, "brier"] == pytest.approx((4 * s10["brier"] + 2 * s11["brier"]) / 6)
    assert pooled.loc[model, "matches"] == 6 and pooled.loc[model, "seasons"] == 2

    rows = preds[preds["forecast"] == model].reset_index(drop=True)
    direct = evaluate(rows["actual"], rows[OUTCOMES])
    assert pooled.loc[model, "accuracy"] == pytest.approx(direct["accuracy"])
    picked = rows[OUTCOMES].idxmax(axis=1)
    assert pooled.loc[model, "macro_f1"] == pytest.approx(
        f1_score(rows["actual"], picked, labels=OUTCOMES, average="macro", zero_division=0))


def test_season_mean_counts_each_season_once_and_differs_from_pooled():
    preds = handmade_predictions()
    seasonal = season_metrics(preds)
    summary = summary_table(preds)
    model = "Logistic Regression"

    mean = (seasonal.loc[(2010, model), "log_loss"] + seasonal.loc[(2011, model), "log_loss"]) / 2
    assert summary.loc[model, "log_loss_season_mean"] == pytest.approx(mean)
    assert summary.loc[model, "log_loss_season_mean"] != pytest.approx(summary.loc[model, "log_loss_pooled"])
    assert summary.loc[model, "log_loss_worst_season"] == seasonal.xs(model, level="forecast")["log_loss"].max()


def test_season_ranks_put_the_lowest_loss_first():
    ranks = season_ranks(handmade_predictions())
    seasonal = season_metrics(handmade_predictions())["log_loss"].unstack("forecast")
    for season in ranks.index:
        assert ranks.loc[season, seasonal.loc[season].idxmin()] == 1


def test_pooled_bootstrap_compares_the_same_matches(backtest):
    table = pooled_bootstrap(backtest.predictions, [(MODELS[0], BOOKMAKER_LABEL), (MODELS[0], BASELINE)],
                             n_resamples=300)
    assert (table["matches"] == 120 + 113 + 120).all()
    pooled = pooled_metrics(backtest.predictions)
    assert table.loc[0, "difference"] == pytest.approx(
        pooled.loc[MODELS[0], "log_loss"] - pooled.loc[BOOKMAKER_LABEL, "log_loss"])


def test_seasonal_bootstrap_has_one_row_per_season_and_pair(backtest):
    pairs = [(MODELS[0], BOOKMAKER_LABEL), (MODELS[1], BOOKMAKER_LABEL)]
    table = seasonal_bootstrap(backtest.predictions, pairs, n_resamples=200)
    seasonal = season_metrics(backtest.predictions)["log_loss"]

    assert len(table) == len(BACKTEST) * len(pairs)
    for row in table.itertuples():
        assert row.difference == pytest.approx(seasonal[(row.season, row.a)] - seasonal[(row.season, row.b)])
    assert table.loc[table["season"] == 2005, "matches"].eq(113).all()


# ---------------------------------------------------------------- artifacts


def test_save_and_load_round_trip(backtest, tmp_path):
    save_backtest(backtest, tmp_path)
    loaded = load_backtest(tmp_path)
    pd.testing.assert_frame_equal(loaded.predictions, backtest.predictions, check_dtype=False)
    assert json.loads(loaded.folds.loc[0, "params"]) == json.loads(backtest.folds.loc[0, "params"])
    with pytest.raises(FileNotFoundError, match="src.backtest"):
        load_backtest(tmp_path / "missing")

"""Tests for recency strategies (src.recency) and their use in the walk-forward backtest."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import src.train as train_module
from src.backtest import (
    BASELINE,
    PREDICTION_COLUMNS,
    common_fixtures,
    pooled_metrics,
    run_backtest,
    run_fold,
    season_metrics,
)
from src.bookmaker import BOOKMAKER_LABEL
from src.evaluate import OUTCOMES
from src.feature_engineering import FEATURE_COLUMNS
from src.recency import (
    CONTROL,
    STRATEGIES,
    RecencyStrategy,
    effective_sample_size,
    exponential_weights,
    match_age_years,
    rolling_window,
    run_recency_backtest,
    season_differences,
    season_level_summary,
    strategy_table,
    training_footprint,
)
from tests.test_backtest import handmade_predictions, make_matches
from tests.test_train import make_features

FAST_LR = [{"C": 0.01}, {"C": 1.0}]
SEASONS = range(2000, 2008)
WINDOW_2 = RecencyStrategy("window_2", "Window: 2 seasons", window_seasons=2)
HALF_LIFE_1 = RecencyStrategy("half_life_1", "Half-life: 1 year", half_life_years=1)


@pytest.fixture(scope="module")
def data():
    features = make_features(seasons=SEASONS)
    return features, make_matches(features)


@pytest.fixture(scope="module")
def experiment(data):
    features, matches = data
    return run_recency_backtest(features, matches, strategies=[CONTROL, WINDOW_2, HALF_LIFE_1],
                                seasons=(2004, 2005, 2006), param_grid=FAST_LR)


# ---------------------------------------------------------------- sample weights


def test_ages_are_measured_from_the_training_cutoff_in_years():
    dates = pd.to_datetime(["2020-05-01", "2019-05-02", "2018-05-01"])
    age = match_age_years(dates)  # cutoff = most recent match
    assert age[0] == 0
    assert age[1] == pytest.approx(365 / 365.25)
    assert age[2] == pytest.approx(731 / 365.25)
    with pytest.raises(ValueError, match="after the training cutoff"):
        match_age_years(dates, cutoff="2019-01-01")


def test_weight_halves_every_half_life_and_is_normalised():
    dates = pd.Timestamp("2024-01-01") - pd.to_timedelta([0, 730.5, 1461], unit="D")  # 0, 2 and 4 years old
    w = exponential_weights(dates, half_life_years=2)

    assert w[1] / w[0] == pytest.approx(0.5)
    assert w[2] / w[0] == pytest.approx(0.25)
    assert w.mean() == pytest.approx(1.0)
    assert (w > 0).all()
    assert np.argmax(w) == 0  # the most recent match weighs most


def test_longer_half_lives_flatten_the_weights():
    dates = pd.date_range("2010-08-01", "2020-05-01", periods=500)
    sizes = [effective_sample_size(exponential_weights(dates, h)) for h in (1, 2, 5, 50)]
    assert sizes == sorted(sizes)
    assert sizes[-1] == pytest.approx(500, rel=0.01)  # very long half-life is almost equal weighting


def test_effective_sample_size():
    assert effective_sample_size(np.ones(10)) == pytest.approx(10)
    assert effective_sample_size([1, 0, 0, 0]) == pytest.approx(1)
    assert effective_sample_size([2, 2, 2]) == pytest.approx(3)  # scale does not matter


@pytest.mark.parametrize("half_life", [0, -1])
def test_invalid_half_life_is_rejected(half_life):
    with pytest.raises(ValueError, match="positive"):
        exponential_weights(pd.to_datetime(["2020-01-01"]), half_life)


# ---------------------------------------------------------------- windows and strategies


def test_rolling_window_keeps_exactly_the_last_k_seasons(data):
    features, _ = data
    train = features[features["season_start"] <= 2005]
    kept = rolling_window(train, 3)
    assert sorted(kept["season_start"].unique()) == [2003, 2004, 2005]
    assert len(kept) == 3 * 120
    assert len(rolling_window(train, 50)) == len(train)  # fewer seasons than the window: keep all
    with pytest.raises(ValueError):
        rolling_window(train, 0)


def test_strategies_prepare_rows_and_weights(data):
    features, _ = data
    train = features[features["season_start"] <= 2005]

    rows, weights = CONTROL.prepare(train)
    assert rows is train and weights is None
    rows, weights = WINDOW_2.prepare(train)
    assert set(rows["season_start"]) == {2004, 2005} and weights is None
    rows, weights = HALF_LIFE_1.prepare(train)
    assert len(rows) == len(train) and len(weights) == len(train)
    assert weights[rows["date"].to_numpy().argmax()] == weights.max()


def test_strategy_definitions_are_validated():
    with pytest.raises(ValueError, match="not both"):
        RecencyStrategy("x", "x", window_seasons=3, half_life_years=2)
    with pytest.raises(ValueError):
        RecencyStrategy("x", "x", window_seasons=0)
    with pytest.raises(ValueError):
        RecencyStrategy("x", "x", half_life_years=-2)
    assert [s.name for s in STRATEGIES] == ["expanding", "window_3", "window_5", "window_10",
                                            "half_life_1", "half_life_2", "half_life_5"]


def test_strategy_table_reports_rows_and_effective_size(data):
    features, _ = data
    table = strategy_table([CONTROL, WINDOW_2, HALF_LIFE_1], features[features["season_start"] <= 2005])
    assert table.loc[CONTROL.label, "rows_used"] == 6 * 120
    assert table.loc[CONTROL.label, "effective_matches"] == pytest.approx(720)
    assert table.loc[WINDOW_2.label, "rows_used"] == 240
    assert table.loc[HALF_LIFE_1.label, "rows_used"] == 720
    assert table.loc[HALF_LIFE_1.label, "effective_matches"] < 720
    assert table.loc[WINDOW_2.label, "weight_of_last_season"] == pytest.approx(0.5)


# ---------------------------------------------------------------- integration with training


def test_the_strategy_is_applied_to_tuning_and_final_fits(data, monkeypatch):
    """Spy on every fit: tuning sees seasons up to S-2, the final fit up to S-1, both windowed."""
    features, _ = data
    seen = []
    original = train_module.training_data

    def spy(train, strategy=None):
        rows, params = original(train, strategy)
        seen.append(sorted(rows["season_start"].unique()))
        return rows, params

    monkeypatch.setattr(train_module, "training_data", spy)
    run_fold(features, 2005, "logistic_regression", FAST_LR, strategy=WINDOW_2)

    assert [2002, 2003] in seen  # inner tuning: train up to 2003, last two seasons
    assert seen[-1] == [2003, 2004]  # final fit: train up to 2004, last two seasons
    assert all(max(s) <= 2004 for s in seen)


def test_sample_weights_reach_the_logistic_regression(data, monkeypatch):
    features, _ = data
    captured = {}
    original = train_module.training_data

    def spy(train, strategy=None):
        rows, params = original(train, strategy)
        captured.update(params)
        return rows, params

    monkeypatch.setattr(train_module, "training_data", spy)
    run_fold(features, 2005, "logistic_regression", FAST_LR, strategy=HALF_LIFE_1)
    assert "model__sample_weight" in captured
    assert captured["model__sample_weight"].mean() == pytest.approx(1.0)


def test_weighting_changes_the_model_but_the_control_does_not(data):
    features, _ = data
    plain = run_fold(features, 2005, "logistic_regression", FAST_LR)
    control = run_fold(features, 2005, "logistic_regression", FAST_LR, strategy=CONTROL)
    weighted = run_fold(features, 2005, "logistic_regression", [{"C": 1.0}], strategy=HALF_LIFE_1)
    unweighted = run_fold(features, 2005, "logistic_regression", [{"C": 1.0}])

    pd.testing.assert_frame_equal(plain.predictions, control.predictions)  # control == commit-10 procedure
    assert not np.allclose(weighted.predictions["prob_home_win"], unweighted.predictions["prob_home_win"])


@pytest.mark.parametrize("strategy", [WINDOW_2, HALF_LIFE_1])
def test_evaluation_season_results_and_later_seasons_never_leak(data, strategy):
    features, _ = data
    changed = features.copy()
    evaluation = changed["season_start"] == 2005
    changed.loc[evaluation, "result"] = np.random.default_rng(4).choice(OUTCOMES, size=int(evaluation.sum()))
    later = changed["season_start"] > 2005
    changed.loc[later, "result"] = "D"
    changed.loc[later, FEATURE_COLUMNS[:10]] = 50.0

    a = run_fold(features, 2005, "logistic_regression", FAST_LR, strategy=strategy)
    b = run_fold(changed, 2005, "logistic_regression", FAST_LR, strategy=strategy)

    assert (a.feature_set, a.params) == (b.feature_set, b.params)
    columns = ["prob_home_win", "prob_draw", "prob_away_win"]
    pd.testing.assert_frame_equal(a.predictions[columns], b.predictions[columns])


# ---------------------------------------------------------------- experiment


def test_every_strategy_is_scored_on_identical_fixtures(experiment):
    preds = experiment.predictions
    labels = {CONTROL.label, WINDOW_2.label, HALF_LIFE_1.label}
    assert set(preds["forecast"]) == labels | {BOOKMAKER_LABEL, BASELINE}
    keys = {label: set(map(tuple, g[["season", "date", "home_team", "away_team"]].astype(str).to_numpy()))
            for label, g in preds[preds["forecast"].isin(labels)].groupby("forecast")}
    assert len({frozenset(k) for k in keys.values()}) == 1
    common_fixtures(preds)  # raises if any forecast differs on fixtures with odds
    assert (season_metrics(preds).loc[2005, "matches"] == 113).all()


def test_control_matches_a_plain_backtest(data, experiment):
    features, matches = data
    plain = run_backtest(features, matches, seasons=(2004, 2005, 2006), models=["logistic_regression"],
                         param_grids={"logistic_regression": FAST_LR})
    a = plain.predictions[plain.predictions["forecast"] == "Logistic Regression"]
    b = experiment.predictions[experiment.predictions["forecast"] == CONTROL.label]
    np.testing.assert_array_equal(a[OUTCOMES].to_numpy(), b[OUTCOMES].to_numpy())


def test_experiment_is_reproducible(data, experiment):
    features, matches = data
    again = run_recency_backtest(features, matches, strategies=[CONTROL, WINDOW_2, HALF_LIFE_1],
                                 seasons=(2004, 2005, 2006), param_grid=FAST_LR)
    pd.testing.assert_frame_equal(again.predictions, experiment.predictions)
    pd.testing.assert_frame_equal(again.folds, experiment.folds)


def test_folds_record_strategy_and_training_footprint(experiment):
    folds = experiment.folds
    assert set(folds["strategy"]) == {"expanding", "window_2", "half_life_1"}
    window = folds[folds["strategy"] == "window_2"]
    assert (window["first_season"] == window["season"] - 2).all()
    assert (window["rows_used"] == 240).all()
    control = folds[folds["strategy"] == "expanding"]
    assert (control["rows_used"] == control["train_matches"]).all()
    half = folds[folds["strategy"] == "half_life_1"]
    assert (half["effective_matches"] < half["rows_used"]).all()


def test_duplicate_strategy_labels_are_rejected(data):
    features, matches = data
    with pytest.raises(ValueError, match="unique"):
        run_recency_backtest(features, matches, strategies=[CONTROL, CONTROL], seasons=(2004,), param_grid=FAST_LR)


# ---------------------------------------------------------------- season-level analysis


def three_season_predictions():
    """A strategy that beats the control in two of three seasons, on hand-made data."""
    base = handmade_predictions()
    control = base[base["forecast"] == "Logistic Regression"].assign(forecast=CONTROL.label)
    third = control[control["season"] == 2010].assign(season=2012)
    control = pd.concat([control, third], ignore_index=True)
    better = control.copy()
    better["forecast"] = "Better"
    sharpen = better["season"].isin([2010, 2012])
    for i in better.index[sharpen]:
        p = better.loc[i, OUTCOMES].astype(float)
        p[better.loc[i, "actual"]] += 0.1
        better.loc[i, OUTCOMES] = (p / p.sum()).to_numpy()
    return pd.concat([control, better], ignore_index=True)


def test_season_differences_are_strategy_minus_control():
    preds = three_season_predictions()
    diffs = season_differences(preds)
    seasonal = season_metrics(preds)["log_loss"].unstack("forecast")
    np.testing.assert_allclose(diffs["Better"], seasonal["Better"] - seasonal[CONTROL.label])
    assert (diffs.loc[[2010, 2012], "Better"] < 0).all()
    assert diffs.loc[2011, "Better"] == pytest.approx(0)


def test_season_level_summary_statistics():
    preds = three_season_predictions()
    summary = season_level_summary(preds).loc["Better"]
    d = season_differences(preds)["Better"]

    assert summary["seasons"] == 3
    assert summary["seasons_better"] == 2 and summary["seasons_worse"] == 0
    assert summary["sign_test_p"] == pytest.approx(stats.binomtest(2, 2, 0.5).pvalue)
    half = stats.t.ppf(0.975, 2) * d.std(ddof=1) / np.sqrt(3)
    assert summary["ci_low"] == pytest.approx(d.mean() - half)
    weights = pd.Series({2010: 4, 2011: 2, 2012: 4})
    assert summary["pooled_difference"] == pytest.approx((d * weights).sum() / 10)
    assert summary["leave_one_out_min"] <= summary["pooled_difference"] <= summary["leave_one_out_max"]


def picks_to_predictions(actual, picks):
    """Long-format predictions where each forecast gives 0.6 to its pick and 0.2 to the others."""
    rows = []
    for forecast, by_season in picks.items():
        for season, season_picks in by_season.items():
            for k, (outcome, pick) in enumerate(zip(actual[season], season_picks)):
                probs = {o: (0.6 if o == pick else 0.2) for o in OUTCOMES}
                rows.append({"season": season, "forecast": forecast, "date": pd.Timestamp(f"{season}-09-{k + 1:02d}"),
                             "home_team": f"T{k}", "away_team": f"U{k}", "actual": outcome, **probs,
                             "has_odds": True})
    return pd.DataFrame(rows)[PREDICTION_COLUMNS]


def test_pooled_macro_f1_difference_comes_from_pooled_predictions():
    """Regression: macro F1 is not an average over matches, so its pooled difference must be
    computed from the pooled predictions, not as a match-weighted mean of seasonal differences."""
    actual = {2010: list("HHHHHHDA"), 2011: list("DDAH")}  # seasons of different size and outcome mix
    preds = picks_to_predictions(actual, {
        CONTROL.label: {2010: list("HHHHHHHH"), 2011: list("HHHH")},
        "Other": {2010: list("HHHHHHDA"), 2011: list("HDAA")},
    })

    summary = season_level_summary(preds, metric="macro_f1").loc["Other"]
    pooled = pooled_metrics(preds)["macro_f1"]
    expected = pooled["Other"] - pooled[CONTROL.label]
    seasonal = season_metrics(preds)
    diffs = season_differences(preds, metric="macro_f1")["Other"]
    sizes = seasonal["matches"].unstack("forecast")[CONTROL.label]
    weighted = float((diffs * sizes).sum() / sizes.sum())

    assert summary["pooled_difference"] == pytest.approx(expected)
    assert abs(expected - weighted) > 0.005  # the two definitions really differ here
    assert summary["pooled_difference"] != pytest.approx(weighted)

    # Leave-one-season-out is pooled the same way: each value is one remaining season's difference
    single = [pooled_metrics(preds[preds["season"] == s])["macro_f1"] for s in (2010, 2011)]
    single = sorted(p["Other"] - p[CONTROL.label] for p in single)
    assert [summary["leave_one_out_min"], summary["leave_one_out_max"]] == pytest.approx(single)

    # Season-level statistics are unchanged: they still use the seasonal differences
    assert summary["mean_difference"] == pytest.approx(diffs.mean())
    assert summary["seasons_better"] == 2 and summary["seasons_worse"] == 0


def test_pooled_difference_still_equals_the_weighted_mean_for_average_metrics():
    actual = {2010: list("HHHHHHDA"), 2011: list("DDAH")}
    preds = picks_to_predictions(actual, {
        CONTROL.label: {2010: list("HHHHHHHH"), 2011: list("HHHH")},
        "Other": {2010: list("HHHHHHDA"), 2011: list("HDAA")},
    })
    sizes = season_metrics(preds)["matches"].unstack("forecast")[CONTROL.label]
    for metric in ("log_loss", "brier", "accuracy"):
        diffs = season_differences(preds, metric=metric)["Other"]
        weighted = float((diffs * sizes).sum() / sizes.sum())
        assert season_level_summary(preds, metric=metric).loc["Other", "pooled_difference"] == pytest.approx(weighted)


def test_higher_is_better_metrics_count_improvements_correctly():
    summary = season_level_summary(three_season_predictions(), metric="accuracy").loc["Better"]
    assert summary["seasons_better"] + summary["seasons_worse"] <= 3
    assert summary["seasons_worse"] == 0


def test_training_footprint_of_the_control_uses_every_row(data):
    features, _ = data
    train = features[features["season_start"] <= 2004]
    footprint = training_footprint(CONTROL, train)
    assert footprint["rows_used"] == len(train)
    assert footprint["weight_of_last_season"] == pytest.approx(120 / len(train))

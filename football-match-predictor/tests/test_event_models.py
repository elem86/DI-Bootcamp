"""Tests for the direct Over/Under and BTTS models (src.event_models) and their backtest (src.event_backtest)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from src.backtest import BACKTEST_SEASONS
from src.event_backtest import (
    CANDIDATES,
    DIRECT,
    EVENTS_DIR,
    FREQUENCY,
    MAIN_METHODS,
    METHODS,
    POISSON_METHODS,
    RECALIBRATED,
    RULE_COMPARISONS,
    calibration_table,
    calibration_window,
    correction_effect,
    crossing_table,
    event_losses,
    fit_calibrators,
    load_event_backtest,
    paired_comparisons,
    pooled_metrics,
    readiness,
    rule_comparisons,
    run_event_backtest,
    save_event_backtest,
    season_metrics,
    season_sensitivity,
)
from src.event_models import (
    EVENT_NAMES,
    OVER_EVENTS,
    DirectEventClassifier,
    EventChoice,
    EventFrequency,
    LogisticCalibrator,
    check_valid,
    crossing_fixtures,
    event_targets,
    jeffreys_rate,
    monotone_over,
    tune_event,
    under_from_over,
)
from src.feature_engineering import build_features
from src.goals_backtest import MODEL, check_identical_fixtures, run_goals_fold, with_probabilities
from src.goals_model import GOALS_FEATURE_SETS
from tests.test_goals_model import FAST as GOAL_GRIDS
from tests.test_goals_model import make_goal_league, scramble

SEASONS = (2007, 2008)
FAST = dict(c_grid=(0.01, 1.0), feature_sets={k: GOALS_FEATURE_SETS[k] for k in ("goals", "compact")},
            goal_grids=GOAL_GRIDS)


@pytest.fixture(scope="module")
def league():
    return make_goal_league()


@pytest.fixture(scope="module")
def features(league):
    return build_features(league)


@pytest.fixture(scope="module")
def backtest(features):
    return run_event_backtest(features, seasons=SEASONS, **FAST)


def by_fixture(predictions, method):
    rows = predictions[predictions["method"] == method]
    return rows.set_index(["date", "home_team", "away_team"])[EVENT_NAMES].sort_index()


# ---------------------------------------------------------------- targets


def test_goal_thresholds_and_btts():
    scores = pd.DataFrame({"home_goals": [0, 1, 1, 2, 2, 3, 0, 4], "away_goals": [0, 0, 1, 0, 1, 1, 3, 0]})
    targets = event_targets(scores)
    assert targets["over_0_5"].tolist() == [0, 1, 1, 1, 1, 1, 1, 1]
    assert targets["over_1_5"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    assert targets["over_2_5"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]  # three or more goals
    assert targets["over_3_5"].tolist() == [0, 0, 0, 0, 0, 1, 0, 1]
    assert targets["btts_yes"].tolist() == [0, 0, 1, 0, 1, 1, 0, 0]  # 0-3 and 4-0 are not BTTS
    with pytest.raises(ValueError, match="no final score"):
        event_targets(pd.DataFrame({"home_goals": pd.array([1, None], dtype="Int64"), "away_goals": [0, 0]}))


def test_event_frequency_is_smoothed_training_share(features):
    train = features[features["season_start"] <= 2004]
    model = EventFrequency().fit(train)
    targets = event_targets(train)
    for event in EVENT_NAMES:
        assert model.rates_[event] == pytest.approx((targets[event].sum() + 0.5) / (len(train) + 1))
    assert jeffreys_rate(0, 10) == pytest.approx(0.5 / 11) and jeffreys_rate(10, 10) == pytest.approx(10.5 / 11)
    prediction = model.predict(features[features["season_start"] == 2006])
    assert (prediction.nunique() == 1).all()


# ---------------------------------------------------------------- monotone Over probabilities


def test_monotone_correction_examples():
    over = pd.DataFrame([[0.9, 0.5, 0.6, 0.2],     # one crossing: 0.5 and 0.6 pooled
                         [0.9, 0.7, 0.4, 0.1],     # already ordered: unchanged
                         [0.6, 0.7, 0.8, 0.1],     # three-way pool
                         [0.3, 0.3, 0.3, 0.3]],    # ties are allowed
                        columns=OVER_EVENTS)
    fixed = monotone_over(over)
    assert fixed.iloc[0].tolist() == pytest.approx([0.9, 0.55, 0.55, 0.2])
    assert fixed.iloc[1].tolist() == over.iloc[1].tolist()
    assert fixed.iloc[2].tolist() == pytest.approx([0.7, 0.7, 0.7, 0.1])
    assert fixed.iloc[3].tolist() == over.iloc[3].tolist()
    assert crossing_fixtures(over).tolist() == [True, False, True, False]
    assert not crossing_fixtures(fixed).any()
    pd.testing.assert_frame_equal(monotone_over(fixed), fixed)  # idempotent
    under = under_from_over(fixed)
    assert np.allclose(under.to_numpy() + fixed.to_numpy(), 1.0)
    with pytest.raises(ValueError):
        monotone_over(pd.DataFrame([[1.2, 0.5, 0.4, 0.1]], columns=OVER_EVENTS))
    with pytest.raises(ValueError):
        monotone_over(pd.DataFrame([[np.nan, 0.5, 0.4, 0.1]], columns=OVER_EVENTS))


def test_monotone_correction_is_the_least_squares_fit():
    rng = np.random.default_rng(3)
    over = pd.DataFrame(rng.uniform(size=(300, 4)), columns=OVER_EVENTS)
    fixed = monotone_over(over).to_numpy()
    reference = np.vstack([IsotonicRegression(increasing=False).fit_transform(np.arange(4), row)
                           for row in over.to_numpy()])
    assert np.allclose(fixed, reference)
    assert (np.diff(fixed, axis=1) <= 1e-15).all() and (fixed >= 0).all() and (fixed <= 1).all()
    assert np.array_equal(fixed, monotone_over(over).to_numpy())  # deterministic


def test_check_valid():
    good = pd.DataFrame([[0.9, 0.7, 0.5, 0.3, 0.5]], columns=EVENT_NAMES)
    check_valid(good)
    with pytest.raises(ValueError, match="increase"):
        check_valid(good.assign(over_2_5=0.8))
    with pytest.raises(ValueError, match="inside"):
        check_valid(good.assign(btts_yes=1.1))


# ---------------------------------------------------------------- models


def test_direct_classifier(features):
    train = features[features["season_start"] <= 2004]
    choices = {e: EventChoice("goals", 0.1) for e in EVENT_NAMES} | {"btts_yes": EventChoice("compact", 1.0)}
    model = DirectEventClassifier(choices).fit(train)
    first_season = features[features["season_start"] == 2000]  # every history feature missing
    raw = model.predict(first_season, monotone=False)
    fixed = model.predict(first_season)
    assert raw.notna().all().all() and ((raw > 0) & (raw < 1)).all().all()
    assert not crossing_fixtures(fixed).any()
    assert (fixed["btts_yes"] == raw["btts_yes"]).all()  # BTTS is never touched by the Over correction
    assert len(model.coefficients("over_2_5")) > len(GOALS_FEATURE_SETS["goals"])  # with missing indicators
    # Logistic Regression with an intercept: mean probability on the training rows = training share
    p = model.predict_event(train, "over_2_5")
    assert p.mean() == pytest.approx(event_targets(train)["over_2_5"].mean(), abs=0.01)


def test_direct_classifier_falls_back_when_training_has_one_outcome(features):
    train = features[features["season_start"] <= 2002].copy()
    train["home_goals"] = 2  # every match has at least 2 goals: Over 0.5 and 1.5 always happen
    model = DirectEventClassifier().fit(train)
    p = model.predict(features[features["season_start"] == 2003])
    assert (p["over_0_5"] == jeffreys_rate(len(train), len(train))).all()


def test_tune_event_scores_on_the_validation_season(features):
    train = features[features["season_start"] <= 2004]
    validation = features[features["season_start"] == 2005]
    table = tune_event(train, validation, "over_2_5", FAST["feature_sets"], (0.01, 1.0))
    assert len(table) == 4 and table["log_loss"].is_monotonic_increasing
    with pytest.raises(ValueError, match="Unknown event"):
        tune_event(train, validation, "over_2", FAST["feature_sets"])


def test_logistic_calibrator_recovers_a_known_distortion():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.2, 0.8, 20_000)
    y = rng.random(truth.size) < truth
    logit = np.log(truth / (1 - truth))
    distorted = 1 / (1 + np.exp(-(2.0 * logit + 0.3)))  # too extreme and shifted
    calibrator = LogisticCalibrator().fit(distorted, y)
    assert calibrator.fitted and calibrator.slope == pytest.approx(0.5, abs=0.05)
    assert calibrator.intercept == pytest.approx(-0.15, abs=0.05)
    assert np.allclose(calibrator.transform(distorted), truth, atol=0.03)
    identity = LogisticCalibrator().fit([0.3, 0.6], [1, 1])  # one outcome only
    assert not identity.fitted and np.allclose(identity.transform([0.2, 0.7]), [0.2, 0.7])


# ---------------------------------------------------------------- calibration chronology


def test_calibrators_use_only_earlier_out_of_sample_seasons(features, backtest):
    assert calibration_window(2018) == [2015, 2016, 2017]
    calibrators = backtest.calibrators
    assert (calibrators["last_calibration_season"] == calibrators["season"] - 1).all()
    assert (calibrators["first_calibration_season"] == calibrators["season"] - 3).all()
    history = {s: with_probabilities(run_goals_fold(features, s, **GOAL_GRIDS).predictions) for s in (2004, 2005, 2006)}
    fitted = fit_calibrators(history, 2007)
    row = calibrators[(calibrators["season"] == 2007) & (calibrators["event"] == "over_2_5")].iloc[0]
    assert fitted["over_2_5"].slope == pytest.approx(row["slope"]) and fitted["over_2_5"].intercept == pytest.approx(row["intercept"])
    assert row["calibration_matches"] == sum((features["season_start"] == s).sum() for s in (2004, 2005, 2006))
    with pytest.raises(ValueError, match="missing"):
        fit_calibrators(history, 2008)  # 2007's out-of-sample predictions are not in this history
    mislabelled = dict(history) | {2006: history[2006].assign(season=2007)}  # rows of the season being calibrated
    with pytest.raises(ValueError, match="evaluated season"):
        fit_calibrators(mislabelled, 2007)


def test_calibration_and_tuning_ignore_the_evaluated_season(features, league):
    season = 2008
    original = run_event_backtest(features, seasons=(season,), **FAST)
    changed = build_features(scramble(league, (league["season_start"] >= season).to_numpy()))
    other = run_event_backtest(changed, seasons=(season,), **FAST)
    pd.testing.assert_frame_equal(original.calibrators, other.calibrators)
    pd.testing.assert_frame_equal(original.tuning, other.tuning)
    first_day = league.loc[league["season_start"] == season, "date"].min()
    for method in METHODS:  # the opening day's forecasts only use earlier seasons
        a, b = by_fixture(original.predictions, method), by_fixture(other.predictions, method)
        day = a.index.get_level_values("date") == first_day
        pd.testing.assert_frame_equal(a[day], b[day])


# ---------------------------------------------------------------- walk-forward folds


def test_folds_are_chronological(backtest, features):
    folds = backtest.folds.set_index("season")
    for season in SEASONS:
        assert folds.loc[season, "last_train_season"] == season - 1
        assert folds.loc[season, "tuning_season"] == season - 1
        assert folds.loc[season, "train_matches"] == int((features["season_start"] < season).sum())
        assert folds.loc[season, "evaluation_matches"] == int((features["season_start"] == season).sum())
        train = features[features["season_start"] < season]
        assert folds.loc[season, "frequency_over_2_5"] == pytest.approx(
            jeffreys_rate(int(event_targets(train)["over_2_5"].sum()), len(train)))
    assert sorted(backtest.predictions["season"].unique()) == list(SEASONS)
    assert set(backtest.tuning["season"]) == set(SEASONS)


def test_same_day_and_future_results_do_not_leak(features, league):
    season = 2008
    day = league.loc[league["season_start"] == season, "date"].unique()[6]
    original = run_event_backtest(features, seasons=(season,), **FAST)
    other = run_event_backtest(build_features(scramble(league, (league["date"] >= day).to_numpy())),
                               seasons=(season,), **FAST)
    for method in METHODS:
        a, b = by_fixture(original.predictions, method), by_fixture(other.predictions, method)
        upto = a.index.get_level_values("date") <= day  # the scrambled day itself included
        pd.testing.assert_frame_equal(a[upto], b[upto])
    later = by_fixture(original.predictions, DIRECT).index.get_level_values("date") > day
    assert not by_fixture(original.predictions, DIRECT)[later].equals(by_fixture(other.predictions, DIRECT)[later])


def test_holdout_seasons_are_refused(features):
    for seasons in ((2025,), (2024, 2026)):
        with pytest.raises(ValueError, match="not allowed"):
            run_event_backtest(features, seasons=seasons, **FAST)
    with pytest.raises(ValueError, match="two earlier seasons"):
        run_event_backtest(features, seasons=(2004,), **FAST)  # calibration would need the 2001 fold


def test_identical_fixtures_and_valid_probabilities(backtest):
    predictions = backtest.predictions
    assert list(dict.fromkeys(predictions["method"])) == METHODS
    check_identical_fixtures(predictions, METHODS)
    check_valid(predictions)
    over = predictions[OVER_EVENTS].to_numpy()
    assert (np.diff(over, axis=1) <= 1e-12).all()
    under = under_from_over(predictions)
    assert np.allclose(under.to_numpy() + predictions[OVER_EVENTS].to_numpy(), 1.0)
    uncorrected = backtest.uncorrected
    assert set(uncorrected["method"]) == set(CANDIDATES)
    check_identical_fixtures(uncorrected, CANDIDATES)


def test_poisson_rows_are_the_commit_15_forecasts(features, backtest):
    fold = with_probabilities(run_goals_fold(features, 2007, **GOAL_GRIDS).predictions)
    for method in POISSON_METHODS:
        expected = fold[fold["method"] == method].set_index(["date", "home_team", "away_team"])[EVENT_NAMES].sort_index()
        mine = by_fixture(backtest.predictions[backtest.predictions["season"] == 2007], method)
        pd.testing.assert_frame_equal(mine, expected, check_exact=False, rtol=1e-12)


def test_reproducible(features, backtest):
    again = run_event_backtest(features, seasons=SEASONS, **FAST)
    pd.testing.assert_frame_equal(again.predictions, backtest.predictions)
    pd.testing.assert_frame_equal(again.calibrators, backtest.calibrators)


# ---------------------------------------------------------------- consistency report and metrics


def test_crossing_report_and_correction_effect(backtest):
    table = crossing_table(backtest)
    assert set(table["method"]) == set(CANDIDATES)
    raw = backtest.uncorrected
    direct = raw[raw["method"] == DIRECT]
    assert table.loc[table["method"] == DIRECT, "crossing"].sum() == crossing_fixtures(direct).sum()
    assert (backtest.folds["crossing_fixtures"].sum() == crossing_fixtures(direct).sum())
    effect = correction_effect(backtest)
    assert len(effect) == len(CANDIDATES) * len(OVER_EVENTS)
    assert (effect.loc[effect["fixtures_changed"] == 0, "difference"] == 0).all()


def test_pooled_metrics_are_match_weighted(backtest):
    predictions = backtest.predictions
    seasonal, pooled = season_metrics(predictions), pooled_metrics(predictions)
    losses = event_losses(predictions)
    for method in METHODS:
        rows = seasonal.xs(method, level="method")
        for metric in ["log_loss_over_2_5", "brier_btts_yes", "mean_over_1_5", "observed_btts_yes"]:
            weighted = (rows[metric] * rows["matches"]).sum() / rows["matches"].sum()
            assert pooled.loc[method, metric] == pytest.approx(weighted)
        assert pooled.loc[method, "log_loss_btts_yes"] == pytest.approx(
            losses.loc[losses["method"] == method, "log_loss_btts_yes"].mean())


def test_comparisons_and_rule(backtest):
    predictions = backtest.predictions
    table = paired_comparisons(predictions, [(DIRECT, FREQUENCY), (RECALIBRATED, MODEL)], n_resamples=1000)
    pooled = pooled_metrics(predictions)
    for row in table.itertuples():
        assert row.difference == pytest.approx(pooled.loc[row.a, row.metric] - pooled.loc[row.b, row.metric])
        assert row.adj_ci_low <= row.ci_low <= row.difference <= row.ci_high <= row.adj_ci_high
    family = rule_comparisons(predictions, n_resamples=1000)
    assert len(family) == RULE_COMPARISONS == 8
    assert family["adjusted_confidence"].iloc[0] == pytest.approx(1 - 0.05 / 8)
    rule = readiness(predictions, n_resamples=1000)
    assert len(rule) == 4 and set(rule["candidate"]) == set(CANDIDATES)
    checks = ["beats_frequency", "competitive_with_poisson", "consistent", "calibrated"]
    assert (rule["ready"] == rule[checks].all(axis=1)).all()
    assert set(rule["best_poisson"]) <= set(POISSON_METHODS)
    sensitivity = season_sensitivity(predictions, [(DIRECT, FREQUENCY)])
    assert (sensitivity["seasons"] == len(SEASONS)).all()
    calibration = calibration_table(predictions)
    assert calibration.loc[calibration["method"] == FREQUENCY, "calibration_slope"].isna().all()
    assert len(calibration) == len(METHODS) * len(EVENT_NAMES)
    assert set(MAIN_METHODS) < set(METHODS)


def test_save_and_load_round_trip(tmp_path, backtest):
    save_event_backtest(backtest, tmp_path)
    loaded = load_event_backtest(tmp_path)
    pd.testing.assert_frame_equal(loaded.predictions, backtest.predictions, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.calibrators, backtest.calibrators, check_dtype=False)
    with pytest.raises(FileNotFoundError):
        load_event_backtest(tmp_path / "nothing")


@pytest.mark.skipif(not (EVENTS_DIR / "predictions.csv").exists(), reason="run python -m src.event_backtest first")
def test_saved_experiment_respects_the_holdout():
    saved = load_event_backtest()
    assert set(saved.predictions["season"]) == set(BACKTEST_SEASONS)
    assert (pd.to_datetime(saved.predictions["date"]) < pd.Timestamp("2025-07-01")).all()
    assert (saved.calibrators["last_calibration_season"] < saved.calibrators["season"]).all()
    check_identical_fixtures(saved.predictions, METHODS)
    check_valid(saved.predictions)

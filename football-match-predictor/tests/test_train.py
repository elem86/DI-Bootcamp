"""Tests for src.train and src.evaluate, using a small synthetic feature table."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import OUTCOMES, evaluate, predict_probabilities, results_table
from src.feature_engineering import FEATURE_COLUMNS, SIDES
from src.train import (
    FEATURE_SETS,
    build_baseline,
    build_model,
    check_feature_columns,
    chronological_split,
    feature_matrix,
    run_experiment,
    target,
)


def make_features(seasons=range(2000, 2006), matches_per_season=120, seed=0):
    """Feature table shaped like features.csv, where form drives the result."""
    rng = np.random.default_rng(seed)
    n = len(seasons) * matches_per_season
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df.insert(0, "season_start", np.repeat(list(seasons), matches_per_season))
    df.insert(1, "date", pd.Timestamp("2000-09-01") + pd.to_timedelta(np.arange(n), unit="D"))
    df.insert(2, "home_team", "Home")
    df.insert(3, "away_team", "Away")
    for side in SIDES:
        df[f"{side}_promoted"] = pd.array(rng.random(n) < 0.15, dtype="boolean")
    df.loc[df["season_start"] == seasons[0], ["home_promoted", "away_promoted"]] = pd.NA

    # Outcome depends on the difference in form, so a model has signal to learn
    strength = df["home_form10_ppg"] - df["away_form10_ppg"] + rng.normal(scale=0.8, size=n)
    df["result"] = np.select([strength > 0.4, strength < -0.4], ["H", "A"], default="D")
    df["result"] = df["result"].astype("string")
    df["home_goals"] = (df["result"] == "H").astype(int)
    df["away_goals"] = (df["result"] == "A").astype(int)

    # Missing history, as in the real data: promoted teams lack last season
    for side in SIDES:
        promoted = df[f"{side}_promoted"].fillna(False).to_numpy(dtype=bool)
        df.loc[promoted, [f"{side}_prev_season_ppg", f"{side}_prev_season_gd"]] = np.nan
    df.loc[df.index[:30], "home_form10_ppg"] = np.nan
    return df


SMALL_GRID = dict(c_grid=(0.01, 1.0))


# ---------------------------------------------------------------- splits

def test_chronological_split_respects_season_boundaries():
    features = make_features()
    train, evaluation = chronological_split(features, train_last_season=2003, eval_season=2004)

    assert train["season_start"].max() == 2003
    assert set(evaluation["season_start"]) == {2004}
    assert train["date"].max() < evaluation["date"].min()
    assert len(train) == 4 * 120  # 2000-2003, every match kept


def test_chronological_split_never_includes_later_seasons_in_training():
    features = make_features()
    train, _ = chronological_split(features, train_last_season=2003, eval_season=2005)

    assert 2004 not in set(train["season_start"])  # gap season not leaked either
    assert 2005 not in set(train["season_start"])


def test_chronological_split_rejects_evaluating_on_the_past():
    with pytest.raises(ValueError, match="must come after"):
        chronological_split(make_features(), train_last_season=2004, eval_season=2004)


def test_chronological_split_skips_unplayed_matches():
    features = make_features()
    features.loc[features.index[-5:], "result"] = pd.NA

    _, evaluation = chronological_split(features, 2004, 2005)
    assert evaluation["result"].notna().all()
    assert len(evaluation) == 115


def test_experiment_seasons_follow_the_plan():
    experiment = run_experiment(make_features(), 2003, 2004, 2005, **SMALL_GRID)

    assert experiment.seasons == {
        "train": [2000, 2003], "validation": 2004, "final_train": [2000, 2004], "test": 2005,
    }
    assert len(experiment.test_results) == 120


# ---------------------------------------------------------------- missing values

def test_missing_values_are_imputed_without_dropping_matches():
    features = make_features()
    columns = FEATURE_SETS["all"]
    X = feature_matrix(features, columns)
    assert X.isna().any().any()  # the synthetic data really has gaps

    model = build_model().fit(X, target(features))
    probs = predict_probabilities(model, X)

    assert len(probs) == len(features)
    assert probs.notna().all().all()


def test_unknown_promoted_flag_becomes_nan_not_false():
    features = make_features()
    X = feature_matrix(features, ["home_promoted"])

    first_season = features["season_start"] == 2000
    assert X.loc[first_season, "home_promoted"].isna().all()
    assert set(X.loc[~first_season, "home_promoted"].unique()) <= {0.0, 1.0}


def test_missing_indicators_keep_missing_history_visible():
    features = make_features()
    columns = FEATURE_SETS["compact"]
    model = build_model().fit(feature_matrix(features, columns), target(features))

    imputer = model.named_steps["impute"]
    with_gaps = {columns[i] for i in imputer.indicator_.features_}
    assert {"home_prev_season_ppg", "away_prev_season_ppg", "home_promoted"} <= with_gaps


def test_imputer_is_fitted_on_training_rows_only():
    features = make_features()
    train, test = chronological_split(features, 2004, 2005)
    columns = FEATURE_SETS["compact"]

    model = build_model().fit(feature_matrix(train, columns), target(train))

    expected = feature_matrix(train, columns).median().to_numpy()
    np.testing.assert_allclose(model.named_steps["impute"].statistics_, expected)


def test_match_with_every_feature_missing_still_gets_a_prediction():
    features = make_features()
    columns = FEATURE_SETS["all"]
    model = build_model().fit(feature_matrix(features, columns), target(features))

    blank = pd.DataFrame(np.nan, index=[0], columns=columns)
    probs = predict_probabilities(model, blank)
    assert probs.notna().all().all()


# ---------------------------------------------------------------- probabilities

@pytest.mark.parametrize("make", [build_model, build_baseline])
def test_probabilities_are_valid(make):
    features = make_features()
    columns = FEATURE_SETS["all"]
    model = make().fit(feature_matrix(features, columns), target(features))

    probs = predict_probabilities(model, feature_matrix(features, columns))

    assert list(probs.columns) == OUTCOMES
    assert ((probs >= 0) & (probs <= 1)).all().all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_results_table_has_one_row_per_match_with_probabilities():
    experiment = run_experiment(make_features(), 2003, 2004, 2005, **SMALL_GRID)
    table = experiment.test_results

    assert list(table.columns) == [
        "date", "home_team", "away_team", "actual", "predicted",
        "prob_home_win", "prob_draw", "prob_away_win", "correct",
    ]
    np.testing.assert_allclose(table[["prob_home_win", "prob_draw", "prob_away_win"]].sum(axis=1), 1.0)
    most_likely = table[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy().argmax(axis=1)
    assert list(table["predicted"]) == [OUTCOMES[i] for i in most_likely]


def test_baseline_always_predicts_the_most_frequent_training_outcome():
    features = make_features()
    baseline = build_baseline().fit(feature_matrix(features, ["home_form5_ppg"]), target(features))
    probs = predict_probabilities(baseline, feature_matrix(features, ["home_form5_ppg"]))

    most_frequent = target(features).value_counts().idxmax()
    assert (probs.idxmax(axis=1) == most_frequent).all()
    shares = target(features).value_counts(normalize=True)
    np.testing.assert_allclose(probs.iloc[0][OUTCOMES], shares.reindex(OUTCOMES).to_numpy())


# ---------------------------------------------------------------- leakage

@pytest.mark.parametrize("column", ["result", "home_goals", "away_goals"])
def test_target_columns_cannot_be_used_as_features(column):
    with pytest.raises(ValueError, match="Target columns"):
        check_feature_columns(["home_form5_ppg", column])


@pytest.mark.parametrize("column", ["home_shots", "b365_home", "season", "date"])
def test_only_engineered_pre_match_features_are_allowed(column):
    with pytest.raises(ValueError, match="pre-match features"):
        check_feature_columns(["home_form5_ppg", column])


def test_every_feature_set_contains_only_pre_match_features():
    for columns in FEATURE_SETS.values():
        check_feature_columns(columns)  # raises if not


def test_test_season_results_do_not_affect_the_final_model():
    original = make_features(seed=3)
    scrambled = original.copy()
    test_rows = scrambled["season_start"] == 2005
    scrambled.loc[test_rows, "result"] = "D"  # change every test-season outcome
    scrambled.loc[test_rows, FEATURE_COLUMNS[:10]] = 99.0  # and some test-season features

    a = run_experiment(original, 2003, 2004, 2005, **SMALL_GRID)
    b = run_experiment(scrambled, 2003, 2004, 2005, **SMALL_GRID)

    assert (a.feature_set, a.C) == (b.feature_set, b.C)  # tuning unaffected
    np.testing.assert_allclose(
        a.model.named_steps["model"].coef_, b.model.named_steps["model"].coef_
    )
    np.testing.assert_allclose(
        a.model.named_steps["impute"].statistics_, b.model.named_steps["impute"].statistics_
    )


# ---------------------------------------------------------------- metrics

def test_evaluate_on_a_hand_checked_example():
    y_true = pd.Series(["H", "H", "D", "A"])
    probs = pd.DataFrame(
        [[0.7, 0.2, 0.1], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]], columns=OUTCOMES
    )

    metrics = evaluate(y_true, probs)

    assert metrics["accuracy"] == pytest.approx(3 / 4)
    assert metrics["per_class_accuracy"] == {"H": 0.5, "D": 1.0, "A": 1.0}
    expected_log_loss = -np.mean(np.log([0.7, 0.2, 0.4, 0.7]))
    assert metrics["log_loss"] == pytest.approx(expected_log_loss)
    matrix = metrics["confusion_matrix"]
    assert list(matrix.index) == list(matrix.columns) == OUTCOMES
    assert matrix.loc["H", "A"] == 1 and matrix.loc["H", "H"] == 1


def test_log_loss_uses_the_right_probability_for_each_outcome():
    """Guards the sklearn label-order pitfall: probability columns are H, D, A."""
    probs = pd.DataFrame([[0.9, 0.05, 0.05]], columns=OUTCOMES)

    confident_and_right = evaluate(pd.Series(["H"]), probs)["log_loss"]
    confident_and_wrong = evaluate(pd.Series(["A"]), probs)["log_loss"]

    assert confident_and_right == pytest.approx(-np.log(0.9))
    assert confident_and_wrong == pytest.approx(-np.log(0.05))


def test_results_table_marks_correct_predictions():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-01", "2026-09-02"]),
        "home_team": ["Inter", "Roma"], "away_team": ["Milan", "Lazio"],
        "result": ["H", "D"],
    })
    probs = pd.DataFrame([[0.6, 0.3, 0.1], [0.5, 0.2, 0.3]], columns=OUTCOMES)

    table = results_table(matches, probs)

    assert list(table["predicted"]) == ["H", "H"]
    assert list(table["correct"]) == [True, False]

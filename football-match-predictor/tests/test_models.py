"""Tests for the Random Forest and XGBoost models (src.models) and multi-model training."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.evaluate import OUTCOMES, predict_probabilities
from src.feature_engineering import FEATURE_COLUMNS
from src.models import MODEL_SPECS, XGBOutcomeClassifier, build_random_forest, build_xgboost, grid
from src.train import (
    BASELINE_LABEL,
    FEATURE_SETS,
    chronological_split,
    comparison_table,
    feature_matrix,
    run_experiment,
    target,
    tune,
)
from tests.test_train import make_features

# Small, fast grids for the tests
FAST_GRIDS = {
    "logistic_regression": grid(C=[0.01, 1.0]),
    "random_forest": grid(n_estimators=[30], max_depth=[4, None], min_samples_leaf=[20]),
    "xgboost": grid(n_estimators=[30], max_depth=[2, 3]),
}
TREE_MODELS = ["random_forest", "xgboost"]


def fitted(model_name, features=None, columns=None, **params):
    features = make_features() if features is None else features
    columns = columns or FEATURE_SETS["all"]
    params = params or FAST_GRIDS[model_name][0]
    model = MODEL_SPECS[model_name].build(**params)
    return model.fit(feature_matrix(features, columns), target(features)), features, columns


# ---------------------------------------------------------------- XGBoost wrapper

def test_xgboost_wrapper_accepts_string_labels():
    X = np.array([[0.0], [1.0], [2.0], [0.1], [1.1], [2.1]])
    y = ["A", "D", "H", "A", "D", "H"]

    model = XGBOutcomeClassifier(n_estimators=5).fit(X, y)

    assert list(model.classes_) == ["A", "D", "H"]
    assert set(model.predict(X)) <= {"A", "D", "H"}
    probs = model.predict_proba(X)
    assert probs.shape == (6, 3)
    assert probs.dtype == np.float64  # XGBoost itself returns float32
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_xgboost_wrapper_is_a_proper_sklearn_estimator():
    model = XGBOutcomeClassifier(max_depth=4, n_estimators=7)
    copy = clone(model)

    assert copy.get_params()["max_depth"] == 4
    assert copy.get_params()["n_estimators"] == 7


def test_xgboost_wrapper_rejects_a_single_outcome():
    with pytest.raises(ValueError, match="two different outcomes"):
        XGBOutcomeClassifier(n_estimators=2).fit(np.zeros((3, 1)), ["H", "H", "H"])


def test_xgboost_handles_missing_values_without_imputation():
    model, features, columns = fitted("xgboost")

    assert list(model.named_steps) == ["model"]  # no imputer in the pipeline
    X = feature_matrix(features, columns)
    assert X.isna().any().any()
    assert predict_probabilities(model, X).notna().all().all()


# ---------------------------------------------------------------- shared behaviour

@pytest.mark.parametrize("model_name", TREE_MODELS)
def test_probabilities_are_valid(model_name):
    model, features, columns = fitted(model_name)
    probs = predict_probabilities(model, feature_matrix(features, columns))

    assert list(probs.columns) == OUTCOMES
    assert ((probs >= 0) & (probs <= 1)).all().all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("model_name", TREE_MODELS)
def test_no_matches_are_dropped_and_blank_rows_still_get_predictions(model_name):
    model, features, columns = fitted(model_name)

    assert len(predict_probabilities(model, feature_matrix(features, columns))) == len(features)
    blank = pd.DataFrame(np.nan, index=[0], columns=columns)
    assert predict_probabilities(model, blank).notna().all().all()


@pytest.mark.parametrize("model_name", TREE_MODELS)
def test_training_is_reproducible(model_name):
    first, features, columns = fitted(model_name)
    second, _, _ = fitted(model_name)
    X = feature_matrix(features, columns)

    pd.testing.assert_frame_equal(predict_probabilities(first, X), predict_probabilities(second, X))


@pytest.mark.parametrize("model_name", TREE_MODELS)
def test_models_learn_the_signal(model_name):
    """On data where form decides the result, a model must beat class frequencies."""
    features = make_features(seed=11)
    train, test = chronological_split(features, 2003, 2005)
    columns = FEATURE_SETS["all"]
    model, _, _ = fitted(model_name, train, columns)

    probs = predict_probabilities(model, feature_matrix(test, columns))
    accuracy = (probs.idxmax(axis=1) == target(test)).mean()
    most_frequent_share = target(test).value_counts(normalize=True).max()
    assert accuracy > most_frequent_share + 0.1


def test_random_forest_imputer_keeps_missing_indicators():
    model, features, columns = fitted("random_forest")
    imputer = model.named_steps["impute"]

    with_gaps = {columns[i] for i in imputer.indicator_.features_}
    assert {"home_prev_season_ppg", "home_promoted"} <= with_gaps


def test_random_forest_imputer_is_fitted_on_training_rows_only():
    features = make_features()
    train, _ = chronological_split(features, 2004, 2005)
    model, _, columns = fitted("random_forest", train)

    expected = feature_matrix(train, columns).median().to_numpy()
    np.testing.assert_allclose(model.named_steps["impute"].statistics_, expected)


# ---------------------------------------------------------------- tuning and experiments

def test_tuning_returns_original_parameter_types():
    """max_depth=None and integers must survive the tuning table unchanged."""
    features = make_features()
    train, validation = chronological_split(features, 2003, 2004)
    candidates = grid(n_estimators=[20], max_depth=[3, None], min_samples_leaf=[20])

    tuning = tune(train, validation, model_name="random_forest", param_grid=candidates)

    assert set(tuning["candidate"]) == {0, 1}
    experiment = run_experiment(features, 2003, 2004, 2005, model_name="random_forest", param_grid=candidates)
    assert experiment.params in candidates
    assert experiment.model.named_steps["model"].max_depth in (3, None)
    assert not isinstance(experiment.model.named_steps["model"].max_depth, float)


def test_c_grid_only_applies_to_logistic_regression():
    with pytest.raises(ValueError, match="c_grid"):
        run_experiment(make_features(), 2003, 2004, 2005, model_name="xgboost", c_grid=(1.0,))


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        run_experiment(make_features(), 2003, 2004, 2005, model_name="neural_net")


@pytest.mark.parametrize("model_name", TREE_MODELS)
def test_test_season_results_do_not_affect_the_final_model(model_name):
    original = make_features(seed=5)
    scrambled = original.copy()
    test_rows = scrambled["season_start"] == 2005
    scrambled.loc[test_rows, "result"] = "D"
    scrambled.loc[test_rows, FEATURE_COLUMNS[:10]] = 99.0

    kwargs = dict(model_name=model_name, param_grid=FAST_GRIDS[model_name])
    a = run_experiment(original, 2003, 2004, 2005, **kwargs)
    b = run_experiment(scrambled, 2003, 2004, 2005, **kwargs)

    assert (a.feature_set, a.params) == (b.feature_set, b.params)
    probe = feature_matrix(original[original["season_start"] == 2005], a.feature_columns)
    pd.testing.assert_frame_equal(predict_probabilities(a.model, probe), predict_probabilities(b.model, probe))


def test_comparison_table_lists_every_model_and_the_baseline_once():
    features = make_features()
    experiments = [
        run_experiment(features, 2003, 2004, 2005, model_name=name, param_grid=FAST_GRIDS[name])
        for name in MODEL_SPECS
    ]

    table = comparison_table(experiments)

    expected_models = [MODEL_SPECS[name].label for name in MODEL_SPECS] + [BASELINE_LABEL]
    assert list(table.loc["validation"].index) == expected_models
    assert list(table.loc["test"].index) == expected_models
    assert (table["matches"].loc["test"] == 120).all()


@pytest.mark.parametrize("model_name", TREE_MODELS)
def test_tree_models_use_the_same_chronological_plan(model_name):
    experiment = run_experiment(
        make_features(), 2003, 2004, 2005, model_name=model_name, param_grid=FAST_GRIDS[model_name]
    )
    assert experiment.seasons == {
        "train": [2000, 2003], "validation": 2004, "final_train": [2000, 2004], "test": 2005,
    }
    assert experiment.label == MODEL_SPECS[model_name].label
    assert set(experiment.test) == {experiment.label, BASELINE_LABEL}


def test_builders_create_independent_pipelines():
    assert build_random_forest() is not build_random_forest()
    assert build_xgboost().named_steps["model"].get_params()["max_depth"] == 3

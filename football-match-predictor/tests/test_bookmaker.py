"""Tests for the bookmaker benchmark (src.bookmaker) and loading saved predictions."""

import numpy as np
import pandas as pd
import pytest

import src.train as train
from src.bookmaker import (
    BOOKMAKER_LABEL,
    ODDS_COLUMNS,
    bookmaker_probabilities,
    build_benchmark,
    complete_odds,
    fixture_odds,
    implied_probabilities,
    odds_coverage,
    overround,
)
from src.evaluate import OUTCOMES, PROBABILITY_COLUMNS
from src.feature_engineering import FEATURE_COLUMNS
from src.train import check_feature_columns, load_predictions, predictions_path


def make_matches():
    """Four fixtures; the last one has no draw odds."""
    return pd.DataFrame(
        {
            "season": ["2025-26"] * 4,
            # datetime64[ns], like src.preprocessing.load_matches
            "date": pd.to_datetime(["2025-08-23", "2025-08-23", "2025-08-24", "2025-08-24"]).astype(
                "datetime64[ns]"
            ),
            "home_team": ["Inter", "Lecce", "Roma", "Milan"],
            "away_team": ["Torino", "Genoa", "Bologna", "Cremonese"],
            "result": ["H", "D", "A", "A"],
            "b365_home": [1.25, 2.60, 2.00, 1.30],
            "b365_draw": [6.50, 3.10, 3.40, np.nan],
            "b365_away": [11.0, 2.90, 4.00, 10.0],
        }
    )


def make_predictions(matches, probs):
    """A saved results table, as written by src.train.save_artifacts."""
    table = matches[["date", "home_team", "away_team"]].copy()
    table["actual"] = matches["result"].to_numpy()
    table["predicted"] = [OUTCOMES[i] for i in np.argmax(probs, axis=1)]
    table[PROBABILITY_COLUMNS] = probs
    table["correct"] = table["actual"] == table["predicted"]
    return table


PROBS_A = np.array([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3], [0.45, 0.3, 0.25], [0.6, 0.25, 0.15]])
PROBS_B = np.array([[0.6, 0.25, 0.15], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4], [0.5, 0.3, 0.2]])


# ---------------------------------------------------------------- odds -> probabilities


def test_implied_probabilities_are_one_over_the_odds():
    implied = implied_probabilities(make_matches())

    assert list(implied.columns) == OUTCOMES
    assert implied.loc[2].tolist() == pytest.approx([1 / 2.0, 1 / 3.4, 1 / 4.0])


def test_overround_is_the_sum_of_implied_probabilities():
    matches = make_matches()
    assert overround(matches).loc[2] == pytest.approx(0.5 + 1 / 3.4 + 0.25)  # 1.044: 4.4 % margin
    assert (overround(matches).dropna() > 1).all()


def test_margin_free_probabilities_sum_to_one_and_keep_the_ratios():
    matches = make_matches()
    probs = bookmaker_probabilities(matches)
    implied = implied_probabilities(matches)

    np.testing.assert_allclose(probs.dropna().sum(axis=1), 1.0)
    # Normalising divides all three by the same total, so ratios are unchanged
    assert probs.loc[2, "H"] / probs.loc[2, "A"] == pytest.approx(implied.loc[2, "H"] / implied.loc[2, "A"])
    assert probs.loc[2].tolist() == pytest.approx([0.479, 0.282, 0.239], abs=1e-3)


def test_outcome_order_follows_the_odds_columns():
    """The shortest price must become the highest probability of the right outcome."""
    probs = bookmaker_probabilities(make_matches())
    assert probs.loc[0].idxmax() == "H"  # Inter at 1.25
    assert ODDS_COLUMNS == {"H": "b365_home", "D": "b365_draw", "A": "b365_away"}

    swapped = make_matches().rename(columns={"b365_home": "b365_away", "b365_away": "b365_home"})
    assert bookmaker_probabilities(swapped).loc[0].idxmax() == "A"


@pytest.mark.parametrize("bad_odds", [np.nan, 1.0, 0.0])
def test_incomplete_or_invalid_odds_give_no_probabilities(bad_odds):
    matches = make_matches()
    matches.loc[0, "b365_away"] = bad_odds

    assert not complete_odds(matches).loc[0]
    assert bookmaker_probabilities(matches).loc[0].isna().all()
    assert np.isnan(overround(matches).loc[0])


def test_odds_coverage_per_season():
    matches = pd.concat([make_matches(), make_matches().assign(season="2026-27")], ignore_index=True)
    matches.loc[4, "result"] = pd.NA  # a 2026-27 fixture not played yet

    coverage = odds_coverage(matches)

    assert coverage.loc["2025-26", "matches"] == 4
    assert coverage.loc["2025-26", "with_odds"] == 3
    assert coverage.loc["2025-26", "coverage"] == pytest.approx(0.75)
    assert coverage.loc["2026-27", "matches"] == 3  # unplayed fixture not counted
    expected = overround(make_matches()).dropna()
    assert coverage.loc["2025-26", "mean_overround"] == pytest.approx(expected.mean())
    assert coverage.loc["2025-26", "max_overround"] == pytest.approx(expected.max())


def test_odds_are_never_accepted_as_model_features():
    assert not set(ODDS_COLUMNS.values()) & set(FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="Not pre-match features"):
        check_feature_columns(["home_form10_ppg", "b365_home"])


# ---------------------------------------------------------------- matching fixtures


def test_fixture_odds_match_the_right_game_whatever_the_row_order():
    matches = make_matches()
    predictions = make_predictions(matches, PROBS_A)
    shuffled = matches.sample(frac=1, random_state=3)

    result = fixture_odds(predictions, shuffled)

    expected = bookmaker_probabilities(matches)
    pd.testing.assert_frame_equal(result[OUTCOMES], expected)
    assert result["has_odds"].tolist() == [True, True, True, False]


def test_fixture_odds_match_dates_stored_at_a_different_resolution():
    matches = make_matches()
    predictions = make_predictions(matches, PROBS_A).astype({"date": "datetime64[s]"})

    result = fixture_odds(predictions, matches)

    assert result["has_odds"].sum() == 3


def test_fixture_odds_reject_an_unknown_fixture():
    matches = make_matches()
    predictions = make_predictions(matches, PROBS_A)
    predictions.loc[1, "home_team"] = "Pisa"

    with pytest.raises(ValueError, match="not found"):
        fixture_odds(predictions, matches)


def test_fixture_odds_reject_a_different_result():
    matches = make_matches()
    predictions = make_predictions(matches, PROBS_A)
    predictions.loc[0, "actual"] = "A"

    with pytest.raises(ValueError, match="different result"):
        fixture_odds(predictions, matches)


def test_fixture_odds_reject_duplicate_fixtures():
    matches = make_matches()
    duplicated = pd.concat([matches, matches.iloc[[0]]], ignore_index=True)
    predictions = make_predictions(matches, PROBS_A)

    with pytest.raises(pd.errors.MergeError):
        fixture_odds(predictions, duplicated)
    with pytest.raises(ValueError, match="more than once"):
        fixture_odds(make_predictions(duplicated, np.vstack([PROBS_A, PROBS_A[:1]])), matches)


# ---------------------------------------------------------------- benchmark set


def test_benchmark_uses_the_same_fixtures_with_complete_odds_for_everyone():
    matches = make_matches()
    a = make_predictions(matches, PROBS_A)
    b = make_predictions(matches, PROBS_B).iloc[::-1]  # stored in a different order

    benchmark = build_benchmark({"A model": a, "B model": b}, matches)

    assert set(benchmark.probabilities) == {"A model", "B model", BOOKMAKER_LABEL}
    assert len(benchmark.fixtures) == 3
    assert benchmark.excluded == 1 and benchmark.total == 4
    assert benchmark.coverage == pytest.approx(0.75)
    assert "Cremonese" not in set(benchmark.fixtures["away_team"])
    for probs in benchmark.probabilities.values():
        assert probs.index.equals(benchmark.fixtures.index)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    # Each model's probabilities stay attached to their own fixture
    roma = benchmark.fixtures.index[benchmark.fixtures["home_team"] == "Roma"][0]
    assert benchmark.probabilities["B model"].loc[roma].tolist() == pytest.approx(PROBS_B[2])
    assert benchmark.probabilities[BOOKMAKER_LABEL].loc[roma, "H"] == pytest.approx(0.479, abs=1e-3)
    assert benchmark.actual.loc[roma] == "A"


def test_benchmark_rejects_models_evaluated_on_different_fixtures():
    matches = make_matches()
    a = make_predictions(matches, PROBS_A)
    b = make_predictions(matches.iloc[:3], PROBS_B[:3])

    with pytest.raises(ValueError, match="same fixtures"):
        build_benchmark({"A model": a, "B model": b}, matches)


# ---------------------------------------------------------------- saved predictions


def test_predictions_path_names(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "PREDICTIONS_DIR", tmp_path)

    assert predictions_path("xgboost", "test").name == "xgboost_test_2026-27.csv"
    assert predictions_path("random_forest", "validation").name == "random_forest_validation_2025-26.csv"
    with pytest.raises(ValueError, match="split"):
        predictions_path("xgboost", "train")


def test_load_predictions_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "PREDICTIONS_DIR", tmp_path)
    saved = make_predictions(make_matches(), PROBS_A)
    saved.to_csv(predictions_path("logistic_regression", "validation"), index=False, date_format="%Y-%m-%d")

    loaded = load_predictions("logistic_regression", "validation")

    assert loaded["date"].dtype == "datetime64[ns]"
    pd.testing.assert_frame_equal(loaded, saved)
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_predictions("xgboost", "validation")

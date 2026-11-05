"""Tests for the probability-quality and uncertainty helpers in src.evaluate."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    OUTCOMES,
    PROBABILITY_COLUMNS,
    bootstrap_table,
    brier_score,
    calibration_summary,
    evaluate,
    metrics_table,
    one_hot,
    paired_bootstrap,
    per_match_brier,
    per_match_log_loss,
    probabilities_from_results,
    reliability_table,
)


def probs_frame(rows, columns=OUTCOMES):
    return pd.DataFrame(rows, columns=columns)


def calibrated_sample(n=20_000, seed=0):
    """Outcomes drawn from the probabilities themselves: perfectly calibrated by construction."""
    rng = np.random.default_rng(seed)
    probs = pd.DataFrame(rng.dirichlet([4, 2.5, 3], size=n), columns=OUTCOMES)
    cumulative = probs.cumsum(axis=1).to_numpy()
    draws = rng.random(n)[:, None]
    actual = pd.Series(np.array(OUTCOMES)[(draws > cumulative).sum(axis=1)])
    return actual, probs


# ---------------------------------------------------------------- Brier score


def test_brier_score_reference_values():
    y = pd.Series(["H", "D", "A"])
    certain_right = probs_frame([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    certain_wrong = probs_frame([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    uniform = probs_frame([[1 / 3] * 3] * 3)

    assert brier_score(y, certain_right) == pytest.approx(0.0)
    assert brier_score(y, certain_wrong) == pytest.approx(2.0)
    assert brier_score(y, uniform) == pytest.approx(2 / 3)


def test_brier_score_matches_a_hand_calculation():
    y = pd.Series(["D"])
    probs = probs_frame([[0.5, 0.3, 0.2]])
    # (0.5 - 0)^2 + (0.3 - 1)^2 + (0.2 - 0)^2
    assert brier_score(y, probs) == pytest.approx(0.25 + 0.49 + 0.04)


def test_brier_score_does_not_depend_on_column_order():
    y = pd.Series(["H", "A"])
    probs = probs_frame([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
    shuffled = probs[["A", "H", "D"]]
    assert brier_score(y, shuffled) == pytest.approx(brier_score(y, probs))


def test_evaluate_and_metrics_table_report_brier():
    y = pd.Series(["H", "D", "A", "H"])
    probs = probs_frame([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]])

    metrics = evaluate(y, probs)
    table = metrics_table({"model": metrics})

    assert metrics["brier"] == pytest.approx(brier_score(y, probs))
    assert table.loc["model", "brier"] == pytest.approx(metrics["brier"])


def test_per_match_losses_average_to_the_overall_metrics():
    actual, probs = calibrated_sample(n=500)
    metrics = evaluate(actual, probs)

    assert per_match_log_loss(actual, probs).mean() == pytest.approx(metrics["log_loss"])
    assert per_match_brier(actual, probs).mean() == pytest.approx(metrics["brier"])


def test_per_match_log_loss_uses_the_probability_of_the_actual_outcome():
    y = pd.Series(["H", "D", "A"])
    probs = probs_frame([[0.5, 0.3, 0.2]] * 3)
    np.testing.assert_allclose(per_match_log_loss(y, probs), -np.log([0.5, 0.3, 0.2]))


def test_one_hot_marks_only_the_actual_outcome():
    hits = one_hot(pd.Series(["A", "H"]))
    assert list(hits.columns) == OUTCOMES
    assert hits.to_numpy().tolist() == [[0, 0, 1], [1, 0, 0]]


@pytest.mark.parametrize(
    "y, probs, message",
    [
        (["H"], probs_frame([[0.5, 0.5]], columns=["H", "D"]), "missing outcome"),
        (["H", "D"], probs_frame([[0.5, 0.3, 0.2]]), "rows"),
        (["X"], probs_frame([[0.5, 0.3, 0.2]]), "Unknown outcomes"),
        (["H"], probs_frame([[np.nan, 0.5, 0.5]]), "missing values"),
    ],
)
def test_probability_metrics_reject_bad_input(y, probs, message):
    with pytest.raises(ValueError, match=message):
        brier_score(pd.Series(y), probs)


def test_saved_results_round_trip_and_are_renormalised():
    results = pd.DataFrame([[0.5, 0.3, 0.2 + 1e-7], [0.1, 0.2, 0.7]], columns=PROBABILITY_COLUMNS)

    probs = probabilities_from_results(results)

    assert list(probs.columns) == OUTCOMES
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-15)
    assert probs.loc[1, "A"] == pytest.approx(0.7)


def test_saved_results_that_do_not_sum_to_one_are_rejected():
    results = pd.DataFrame([[0.5, 0.3, 0.3]], columns=PROBABILITY_COLUMNS)
    with pytest.raises(ValueError, match="sum to 1"):
        probabilities_from_results(results)


# ---------------------------------------------------------------- calibration


@pytest.mark.parametrize("strategy", ["quantile", "uniform"])
def test_reliability_table_counts_every_match_once(strategy):
    actual, probs = calibrated_sample(n=1000)
    table = reliability_table(actual, probs, "D", n_bins=5, strategy=strategy)

    assert table["matches"].sum() == 1000
    assert (table["lowest"] <= table["mean_predicted"]).all()
    assert (table["mean_predicted"] <= table["highest"]).all()
    assert ((table["ci_low"] <= table["observed"]) & (table["observed"] <= table["ci_high"])).all()


def test_quantile_bins_have_equal_size_even_with_ties():
    actual = pd.Series(["H", "D", "A", "H"] * 25)
    probs = probs_frame([[0.4, 0.3, 0.3]] * 100)  # every probability tied

    table = reliability_table(actual, probs, "H", n_bins=4, strategy="quantile")

    assert table["matches"].tolist() == [25, 25, 25, 25]


def test_uniform_bins_include_probabilities_of_exactly_zero_and_one():
    actual = pd.Series(["H", "A"])
    probs = probs_frame([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    table = reliability_table(actual, probs, "H", n_bins=10, strategy="uniform")

    assert table["matches"].tolist() == [1, 1]
    assert table["observed"].tolist() == [0.0, 1.0]


def test_a_calibrated_forecast_is_close_to_the_diagonal():
    actual, probs = calibrated_sample()
    for outcome in OUTCOMES:
        table = reliability_table(actual, probs, outcome, n_bins=10)
        assert (table["observed"] - table["mean_predicted"]).abs().max() < 0.04


def test_an_overconfident_forecast_is_flagged():
    actual, probs = calibrated_sample()
    # Push every forecast towards its favourite: overconfident
    sharpened = probs**3
    sharpened = sharpened.div(sharpened.sum(axis=1), axis=0)

    honest = calibration_summary(actual, probs)
    overconfident = calibration_summary(actual, sharpened)

    assert (overconfident["ece"] > honest["ece"] + 0.03).all()


def test_calibration_summary_reports_rates_and_one_vs_rest_brier():
    y = pd.Series(["H", "H", "D", "A"])
    probs = probs_frame([[0.5, 0.3, 0.2]] * 4)

    summary = calibration_summary(y, probs, n_bins=2)

    assert summary.loc["H", "observed"] == pytest.approx(0.5)
    assert summary.loc["D", "mean_predicted"] == pytest.approx(0.3)
    assert summary.loc["A", "difference"] == pytest.approx(0.2 - 0.25)
    assert summary["brier"].sum() == pytest.approx(brier_score(y, probs))


def test_reliability_table_rejects_unknown_outcome_and_strategy():
    actual, probs = calibrated_sample(n=50)
    with pytest.raises(ValueError, match="Unknown outcome"):
        reliability_table(actual, probs, "X")
    with pytest.raises(ValueError, match="strategy"):
        reliability_table(actual, probs, "H", strategy="kmeans")


# ---------------------------------------------------------------- bootstrap


def test_identical_forecasts_have_zero_difference():
    actual, probs = calibrated_sample(n=300)
    result = paired_bootstrap(actual, probs, probs.copy(), n_resamples=500)

    assert result["difference"] == 0
    assert result["ci_low"] == result["ci_high"] == 0
    assert result["share_a_better"] == 0


def test_a_clearly_better_forecast_has_an_interval_below_zero():
    actual, probs = calibrated_sample(n=2000)
    uniform = probs_frame([[1 / 3] * 3] * len(probs))

    result = paired_bootstrap(actual, probs, uniform, n_resamples=2000)

    assert result["difference"] < 0
    assert result["ci_high"] < 0
    assert result["share_a_better"] == pytest.approx(1.0)


def test_bootstrap_is_reproducible_and_antisymmetric():
    actual, probs = calibrated_sample(n=200)
    other = (probs + 0.1).div((probs + 0.1).sum(axis=1), axis=0)

    first = paired_bootstrap(actual, probs, other, n_resamples=1500)
    again = paired_bootstrap(actual, probs, other, n_resamples=1500)
    swapped = paired_bootstrap(actual, other, probs, n_resamples=1500)

    assert first == again
    assert swapped["difference"] == pytest.approx(-first["difference"])
    assert swapped["ci_low"] == pytest.approx(-first["ci_high"])
    assert first["ci_low"] <= first["difference"] <= first["ci_high"]


def test_bootstrap_interval_widens_with_fewer_matches():
    actual, probs = calibrated_sample(n=4000)
    other = (probs + 0.2).div((probs + 0.2).sum(axis=1), axis=0)

    def width(n):
        r = paired_bootstrap(actual[:n], probs[:n], other[:n], n_resamples=2000)
        return r["ci_high"] - r["ci_low"]

    assert width(100) > 2 * width(1600)


def test_bootstrap_supports_brier_and_rejects_unknown_metrics():
    actual, probs = calibrated_sample(n=100)
    uniform = probs_frame([[1 / 3] * 3] * 100)

    result = paired_bootstrap(actual, probs, uniform, metric="brier", n_resamples=200)
    assert result["difference"] == pytest.approx(brier_score(actual, probs) - brier_score(actual, uniform))
    with pytest.raises(ValueError, match="metric"):
        paired_bootstrap(actual, probs, uniform, metric="accuracy")


def test_bootstrap_requires_the_same_matches():
    actual, probs = calibrated_sample(n=50)
    with pytest.raises(ValueError, match="same matches"):
        paired_bootstrap(actual, probs, probs.iloc[::-1])


def test_bootstrap_table_has_one_row_per_pair():
    actual, probs = calibrated_sample(n=300)
    uniform = probs_frame([[1 / 3] * 3] * 300)
    forecasts = {"model": probs, "uniform": uniform, "copy": probs.copy()}

    table = bootstrap_table(actual, forecasts, [("model", "uniform"), ("model", "copy")], n_resamples=300)

    assert table[["a", "b"]].values.tolist() == [["model", "uniform"], ["model", "copy"]]
    assert table["ci_includes_zero"].tolist() == [False, True]

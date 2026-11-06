"""Tests for the adaptive scoring-rate methods (src.scoring_drift) and their backtest (src.scoring_drift_backtest)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BACKTEST_SEASONS
from src.event_backtest import EVENTS_DIR, FREQUENCY, PREDICTION_COLUMNS, RECALIBRATED, load_event_backtest
from src.event_backtest import METHODS as EXISTING_METHODS
from src.event_models import EVENT_NAMES, OVER_EVENTS, EventFrequency, check_valid, crossing_fixtures, event_targets
from src.goals_backtest import LEAGUE, MODEL, TEAM, check_identical_fixtures
from src.goals_model import FIXTURE_KEYS, LAMBDA_COLUMNS, LeagueAverageGoals, total_goals_probabilities
from src.scoring_drift import (
    AdaptiveEventFrequency,
    AdaptiveScoringLevel,
    completed_history,
    level_factor,
    poisson_events,
    recent_window,
    response_share,
    shrunk_mean,
)
from src.scoring_drift_backtest import (
    ADAPTIVE_FREQUENCY,
    ADAPTIVE_POISSON,
    CANDIDATES,
    DRIFT_DIR,
    METHODS,
    RULE_COMPARISONS,
    block_errors,
    error_persistence,
    load_drift_backtest,
    phase_table,
    readiness,
    rule_comparisons,
    run_scoring_drift_backtest,
    save_drift_backtest,
    season_bias,
    season_goal_levels,
    season_phase,
    strongest_existing,
)
from tests.test_goals_model import make_goal_league, scramble

SEASONS = (2007, 2008)
SMALL = dict(window=30, prior_matches=20.0)  # 30 is not a multiple of the 4 matches per date


@pytest.fixture(scope="module")
def league():
    return make_goal_league()


def fake_existing(matches, seasons=SEASONS, seed=3):
    """Commit-15 style rates and commit-16 style event forecasts built quickly from the match table.

    The static frequency and league-average rates are the real ones for each
    season's training rows; the model rates are random but valid.
    """
    rng = np.random.default_rng(seed)
    goal_rows, event_rows = [], []
    for season in seasons:
        train = matches[matches["season_start"] <= season - 1]
        evaluation = matches[matches["season_start"] == season].sort_values(FIXTURE_KEYS)
        base = evaluation[FIXTURE_KEYS + ["home_goals", "away_goals"]].astype({"home_goals": "int64",
                                                                               "away_goals": "int64"})
        base = base.reset_index(drop=True).assign(season=season, promoted_club=False)
        league_rates = LeagueAverageGoals().fit(train).predict(base)
        model_rates = league_rates * rng.lognormal(0, 0.25, size=(len(base), 2))
        team_rates = league_rates * rng.lognormal(0, 0.3, size=(len(base), 2))
        for method, rates in ((LEAGUE, league_rates), (MODEL, model_rates)):
            goal_rows.append(pd.concat([base, rates.reset_index(drop=True)], axis=1).assign(method=method))
        probabilities = {
            FREQUENCY: EventFrequency().fit(train).predict(base),
            LEAGUE: poisson_events(league_rates),
            TEAM: poisson_events(team_rates),
            MODEL: poisson_events(model_rates),
            "Direct logistic": poisson_events(league_rates * rng.lognormal(0, 0.1, size=(len(base), 2))),
            RECALIBRATED: poisson_events(model_rates * 1.05),
        }
        for method, table in probabilities.items():
            frame = base.assign(method=method)
            frame[EVENT_NAMES] = table[EVENT_NAMES].to_numpy()
            event_rows.append(frame[PREDICTION_COLUMNS])
    return pd.concat(goal_rows, ignore_index=True), pd.concat(event_rows, ignore_index=True)


@pytest.fixture(scope="module")
def existing(league):
    return fake_existing(league)


@pytest.fixture(scope="module")
def backtest(league, existing):
    goals, events = existing
    return run_scoring_drift_backtest(league, goals, events, seasons=SEASONS, **SMALL)


def by_fixture(predictions, method):
    rows = predictions[predictions["method"] == method]
    return rows.set_index(FIXTURE_KEYS)[EVENT_NAMES].sort_index()


# ---------------------------------------------------------------- the online window


def test_window_holds_the_last_matches_strictly_before_each_date(league):
    history = completed_history(league)
    dates = pd.Series(sorted(history["date"].unique())[3:60:7])
    table = recent_window(history, dates, window=30)
    for date, row in zip(dates, table.itertuples()):
        earlier = history[history["date"] < date].tail(30)  # same chronological order as the window
        assert row.window_matches == len(earlier)
        assert row.window_home_goals == earlier["home_goals"].sum()
        assert row.window_away_goals == earlier["away_goals"].sum()
        assert row.window_over_2_5 == ((earlier["home_goals"] + earlier["away_goals"]) > 2.5).sum()
        assert row.window_btts_yes == ((earlier["home_goals"] > 0) & (earlier["away_goals"] > 0)).sum()
        assert row.window_last_date < date


def test_window_before_any_match_is_empty(league):
    history = completed_history(league)
    table = recent_window(history, [history["date"].min(), history["date"].min() - pd.Timedelta(days=30)])
    assert (table["window_matches"] == 0).all() and (table["window_home_goals"] == 0).all()
    assert table["window_first_date"].isna().all()
    with pytest.raises(ValueError):
        recent_window(history.iloc[::-1], [history["date"].max()])


def test_validation_and_holdout_seasons_never_enter_the_history(league):
    leak = league[league["season_start"] == 2005].copy()
    leak["season_start"] = 2025  # dated in 2005, but labelled 2025-26: must still be dropped
    history = completed_history(pd.concat([league, leak], ignore_index=True))
    assert history["season_start"].max() < 2025
    assert len(history) == len(completed_history(league))


def test_same_day_and_later_scores_do_not_change_a_forecast_but_earlier_ones_do(league):
    history = completed_history(league)
    date = sorted(history.loc[history["season_start"] == 2008, "date"].unique())[5]
    fixtures = history[history["date"] == date]
    train = history[history["season_start"] <= 2007]

    def forecast(matches):
        visible = completed_history(matches)
        frequency = AdaptiveEventFrequency(**SMALL).fit(train).predict(fixtures, visible)
        level = AdaptiveScoringLevel(**SMALL).fit(train)
        rates = pd.DataFrame({"lambda_home": 1.4, "lambda_away": 1.1}, index=fixtures.index)
        return frequency, level.adjust(rates, fixtures, visible)

    before = forecast(league)
    same_or_later = forecast(scramble(league, league["date"] >= date))
    pd.testing.assert_frame_equal(before[0], same_or_later[0])
    pd.testing.assert_frame_equal(before[1], same_or_later[1])
    earlier_in_season = (league["season_start"] == 2008) & (league["date"] < date)
    changed = forecast(scramble(league, earlier_in_season, seed=7))
    assert not np.allclose(before[0].to_numpy(), changed[0].to_numpy())  # the online update uses them
    assert not np.allclose(before[1].to_numpy(), changed[1].to_numpy())


# ---------------------------------------------------------------- formulas


def test_shrinkage_and_factor_formulas():
    assert shrunk_mean(60, 20, 2.0, 20) == pytest.approx((60 + 40) / 40)
    assert level_factor([60], [20], 2.0, 20)[0] == pytest.approx(1.25)
    assert level_factor([0], [0], 2.0, 20)[0] == 1.0  # empty window: unadjusted
    for bad in (0.0, -1.0, np.nan, np.inf):  # no division by zero or by an invalid mean
        assert level_factor([10], [5], bad, 20)[0] == 1.0
    with pytest.raises(ValueError):
        shrunk_mean(1, 1, 1.0, 0)
    assert response_share([0, 100, 200, 400]) == pytest.approx([0, 0.25, 0.5, 0.5])


def test_adaptive_frequency_formula_and_prior(league):
    history = completed_history(league)
    train = history[history["season_start"] <= 2007]
    fixtures = history[history["season_start"] == 2008]
    model = AdaptiveEventFrequency(**SMALL).fit(train)
    static = EventFrequency().fit(train).rates_
    assert model.prior_ == static
    window = model.components(fixtures, history)
    forecast = model.predict(fixtures, history)
    for event in EVENT_NAMES:
        expected = (window[f"window_{event}"] + 20 * static[event]) / (window["window_matches"] + 20)
        np.testing.assert_allclose(forecast[event], expected, rtol=1e-12)
    # no earlier matches at all: the static frequency
    first = history[history["date"] == history["date"].min()]
    early = model.predict(first, history)
    for event in EVENT_NAMES:
        np.testing.assert_allclose(early[event], static[event], rtol=1e-12)


def test_adaptive_frequency_is_valid_and_monotone(league):
    history = completed_history(league)
    model = AdaptiveEventFrequency(window=5, prior_matches=0.5).fit(history[history["season_start"] <= 2003])
    forecast = model.predict(history[history["season_start"] >= 2004], history)
    check_valid(forecast)
    assert not crossing_fixtures(forecast).any()
    assert ((forecast > 0) & (forecast < 1)).all().all()


def test_adaptive_poisson_rescales_rates_and_reuses_poisson_formulas(league):
    history = completed_history(league)
    train = history[history["season_start"] <= 2007]
    fixtures = history[history["season_start"] == 2008].reset_index(drop=True)
    level = AdaptiveScoringLevel(**SMALL).fit(train)
    assert level.mu_home_ == pytest.approx(train["home_goals"].mean())
    rates = pd.DataFrame({"lambda_home": np.linspace(0.8, 2.4, len(fixtures)),
                          "lambda_away": np.linspace(1.5, 0.6, len(fixtures))})
    factors = level.factors(fixtures, history)
    adjusted = level.adjust(rates, fixtures, history)
    n = factors["window_matches"]
    expected_home = (factors["recent_home"] * n + 20 * level.mu_home_) / (n + 20) / level.mu_home_
    np.testing.assert_allclose(factors["home_factor"], expected_home, rtol=1e-12)
    np.testing.assert_allclose(adjusted["lambda_home"], rates["lambda_home"] * factors["home_factor"], rtol=1e-12)
    np.testing.assert_allclose(adjusted["lambda_away"], rates["lambda_away"] * factors["away_factor"], rtol=1e-12)
    forecast = level.predict(rates, fixtures, history)
    exact = total_goals_probabilities(adjusted["lambda_home"], adjusted["lambda_away"])
    np.testing.assert_allclose(forecast[OVER_EVENTS], exact[OVER_EVENTS], rtol=1e-12)
    btts = (1 - np.exp(-adjusted["lambda_home"])) * (1 - np.exp(-adjusted["lambda_away"]))
    np.testing.assert_allclose(forecast["btts_yes"], btts, rtol=1e-12)
    check_valid(forecast)


def test_adaptive_poisson_without_recent_matches_is_unadjusted(league):
    history = completed_history(league)
    first = history[history["date"] == history["date"].min()]
    level = AdaptiveScoringLevel().fit(history[history["season_start"] <= 2003])
    rates = pd.DataFrame({"lambda_home": [1.3] * len(first), "lambda_away": [0.9] * len(first)}, index=first.index)
    pd.testing.assert_frame_equal(level.predict(rates, first, history), poisson_events(rates))


def test_adaptive_poisson_refuses_invalid_rates(league):
    history = completed_history(league)
    fixtures = history[history["season_start"] == 2008].head(3)
    level = AdaptiveScoringLevel().fit(history[history["season_start"] <= 2007])
    for bad in (np.nan, -0.5, np.inf):
        rates = pd.DataFrame({"lambda_home": [1.0, bad, 1.0], "lambda_away": [1.0, 1.0, 1.0]}, index=fixtures.index)
        with pytest.raises(ValueError):
            level.adjust(rates, fixtures, history)
    zero = pd.DataFrame({"lambda_home": [0.0] * 3, "lambda_away": [0.0] * 3}, index=fixtures.index)
    assert (level.adjust(zero, fixtures, history) == 0).all().all()


# ---------------------------------------------------------------- the backtest


def test_every_method_predicts_identical_fixtures(backtest):
    predictions = backtest.predictions
    assert list(dict.fromkeys(predictions["method"])) == METHODS
    check_identical_fixtures(predictions, METHODS)
    check_valid(predictions)
    assert set(predictions["season"]) == set(SEASONS)


def test_existing_forecasts_are_passed_through_unchanged(backtest, existing):
    _, events = existing
    for method in EXISTING_METHODS:
        pd.testing.assert_frame_equal(by_fixture(backtest.predictions, method), by_fixture(events, method))


def test_fitting_uses_earlier_seasons_and_the_window_may_use_earlier_fixtures_of_the_season(backtest, league):
    folds = backtest.folds.set_index("season")
    drift = backtest.drift
    for season in SEASONS:
        train = league[league["season_start"] <= season - 1]
        assert folds.loc[season, "train_home_goals"] == pytest.approx(train["home_goals"].astype(float).mean())
        assert folds.loc[season, "train_away_goals"] == pytest.approx(train["away_goals"].astype(float).mean())
        rows = drift[drift["season"] == season]
        assert (rows["window_last_date"] < rows["date"]).all()  # never the same date or later
        assert (rows["window_matches"] == SMALL["window"]).all()
        assert (rows["window_last_date"].iloc[-1] >= rows["date"].min())  # online update inside season S
        assert folds.loc[season, "window_from_season"] == season - 1  # the first window reaches back


def test_backtest_matches_the_formulas(backtest):
    drift = backtest.drift
    adaptive = by_fixture(backtest.predictions, ADAPTIVE_POISSON)
    rates = drift.set_index(FIXTURE_KEYS).sort_index()
    np.testing.assert_allclose(rates["lambda_home_adjusted"], rates["lambda_home"] * rates["home_factor"])
    expected = poisson_events(rates[["lambda_home_adjusted", "lambda_away_adjusted"]].set_axis(LAMBDA_COLUMNS,
                                                                                                axis=1))
    np.testing.assert_allclose(adaptive.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_backtest_is_reproducible_and_ignores_row_order(backtest, league, existing):
    goals, events = existing
    again = run_scoring_drift_backtest(league.sample(frac=1, random_state=1), goals.sample(frac=1, random_state=2),
                                       events.sample(frac=1, random_state=3), seasons=SEASONS, **SMALL)
    for method in CANDIDATES:
        pd.testing.assert_frame_equal(by_fixture(again.predictions, method), by_fixture(backtest.predictions, method))


def test_backtest_refuses_holdout_seasons_and_inconsistent_inputs(league, existing):
    goals, events = existing
    with pytest.raises(ValueError):
        run_scoring_drift_backtest(league, goals, events, seasons=[2025], **SMALL)
    with pytest.raises(ValueError, match="lack"):
        run_scoring_drift_backtest(league, goals, events[events["method"] != TEAM], seasons=SEASONS, **SMALL)
    wrong_static = events.copy()
    wrong_static.loc[wrong_static["method"] == FREQUENCY, "over_2_5"] += 0.01
    with pytest.raises(ValueError, match="static"):
        run_scoring_drift_backtest(league, goals, wrong_static, seasons=SEASONS, **SMALL)
    wrong_rates = goals.copy()
    wrong_rates.loc[wrong_rates["method"] == MODEL, "lambda_home"] *= 1.1
    with pytest.raises(ValueError, match="reproduce"):
        run_scoring_drift_backtest(league, wrong_rates, events, seasons=SEASONS, **SMALL)
    wrong_scores = scramble(league, league["season_start"] == 2008, seed=5)
    with pytest.raises(ValueError, match="Scores"):
        run_scoring_drift_backtest(wrong_scores, goals, events, seasons=SEASONS, **SMALL)


def test_decision_rule_and_comparisons(backtest):
    predictions = backtest.predictions
    family = rule_comparisons(predictions, n_resamples=500)
    assert len(family) == RULE_COMPARISONS == 8
    np.testing.assert_allclose(family["adjusted_confidence"], 1 - 0.05 / 8)
    assert set(family["a"]) == set(CANDIDATES)
    for event in ("over_2_5", "btts_yes"):
        assert strongest_existing(predictions, event) not in (FREQUENCY, *CANDIDATES)
    table = readiness(predictions, n_resamples=500)
    assert len(table) == 4
    checks = ["beats_static", "competitive", "five_of_seven", "leave_one_out", "calibrated"]
    assert (table["ready"] == table[checks].all(axis=1)).all()


def test_drift_diagnostics(backtest, league):
    levels = season_goal_levels(league)
    assert levels.loc[2008, "home_goals"] == pytest.approx(league.loc[league["season_start"] == 2008,
                                                                      "home_goals"].astype(float).mean())
    phase = season_phase(backtest.predictions)
    counts = backtest.predictions.assign(phase=phase).groupby(["season", "method", "phase"]).size()
    assert counts.unstack("phase").nunique().max() == 1 and set(phase) == {0, 1}
    early = backtest.predictions[phase == 0].groupby(["season", "method"])["date"].max()
    late = backtest.predictions[phase == 1].groupby(["season", "method"])["date"].min()
    assert (early <= late).all()
    assert phase_table(backtest.predictions).shape == (len(METHODS), 4)
    blocks = block_errors(backtest.drift, blocks_per_season=4)
    assert len(blocks) == 4 * len(SEASONS)
    persistence = error_persistence(blocks, max_lag=2)
    assert persistence.loc[0, "error_vs_earlier_adjustment"] == pytest.approx(blocks["error"].corr(blocks["adjustment"]))
    bias = season_bias(backtest.predictions)
    targets = event_targets(backtest.predictions)
    rows = (backtest.predictions["method"] == FREQUENCY) & (backtest.predictions["season"] == 2008)
    expected = backtest.predictions.loc[rows, "over_2_5"].mean() - targets.loc[rows, "over_2_5"].mean()
    assert bias.loc[("over_2_5", 2008), FREQUENCY] == pytest.approx(expected)


def test_save_and_load(backtest, tmp_path):
    save_drift_backtest(backtest, tmp_path)
    loaded = load_drift_backtest(tmp_path)
    for method in METHODS:
        pd.testing.assert_frame_equal(by_fixture(loaded.predictions, method), by_fixture(backtest.predictions, method))
    assert (loaded.drift["window_last_date"] < loaded.drift["date"]).all()
    assert (tmp_path / "settings.json").exists()


@pytest.mark.skipif(not (DRIFT_DIR / "predictions.csv").exists() or not (EVENTS_DIR / "predictions.csv").exists(),
                    reason="run python -m src.scoring_drift_backtest first")
def test_saved_experiment_respects_the_chronology():
    saved = load_drift_backtest()
    assert set(saved.predictions["season"]) == set(BACKTEST_SEASONS)
    assert (saved.predictions["date"] < pd.Timestamp("2025-07-01")).all()
    assert (saved.drift["window_last_date"] < saved.drift["date"]).all()
    assert (saved.drift["window_matches"] == 200).all()
    check_identical_fixtures(saved.predictions, METHODS)
    check_valid(saved.predictions)
    events = load_event_backtest().predictions
    for method in EXISTING_METHODS:
        np.testing.assert_array_equal(by_fixture(saved.predictions, method).to_numpy(),
                                      by_fixture(events, method).to_numpy())

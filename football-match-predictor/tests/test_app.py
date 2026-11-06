"""Tests for the app's layout, data-freshness warnings and research summaries (app.py, src/app_support.py)."""

import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.app_support as support
from src.app_support import (
    ARTIFACTS,
    DECISIONS,
    STALE_AFTER_DAYS,
    ResearchFileMissing,
    artifact_path,
    available_artifacts,
    data_freshness,
)
from src.config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR

PROJECT = Path(__file__).resolve().parents[1]
HAS_APP_INPUTS = (PROCESSED_DATA_DIR / "matches.csv").exists() and all(
    (MODELS_DIR / f"{name}.joblib").exists() for name in ("logistic_regression", "random_forest", "xgboost"))
HAS_RESEARCH = all(available_artifacts(OUTPUTS_DIR).values())
TABS = ["Match predictions", "Goals & BTTS history", "Research & methodology"]


# ---------------------------------------------------------------- freshness


def test_fresh_data_has_no_warnings():
    fresh = data_freshness("2026-11-02", "2026-11-07", today_date="2026-11-06")
    assert fresh.data_age_days == 4 and fresh.fixture_gap_days == 5
    assert not fresh.data_is_stale and not fresh.fixture_is_far_ahead
    assert fresh.warnings() == []


def test_stale_data_and_distant_fixtures_are_warned_about():
    latest = dt.date(2026, 11, 2)
    boundary = data_freshness(latest, latest + dt.timedelta(days=STALE_AFTER_DAYS),
                              today_date=latest + dt.timedelta(days=STALE_AFTER_DAYS))
    assert boundary.warnings() == []  # exactly the threshold is still acceptable
    stale = data_freshness(latest, latest + dt.timedelta(days=40), today_date=latest + dt.timedelta(days=30))
    messages = stale.warnings()
    assert len(messages) == 2
    assert "30 days ago" in messages[0] and "02 Nov 2026" in messages[0]
    assert "40 days after" in messages[1]
    only_far = data_freshness(latest, latest + dt.timedelta(days=60), today_date=latest + dt.timedelta(days=1))
    assert only_far.warnings() == [only_far.warnings()[0]] and "60 days after" in only_far.warnings()[0]


def test_today_is_used_by_default(monkeypatch):
    monkeypatch.setattr(support, "today", lambda: dt.date(2027, 1, 1))
    assert data_freshness("2026-12-01", "2027-01-05").data_age_days == 31


# ---------------------------------------------------------------- research artifacts


def test_missing_artifacts_raise_an_explanation(tmp_path):
    assert not any(available_artifacts(tmp_path).values())
    for name, (_, command, _) in ARTIFACTS.items():
        with pytest.raises(ResearchFileMissing) as error:
            artifact_path(tmp_path, name)
        assert error.value.command == command and command in str(error.value)
    for loader in (support.outcome_model_summary, support.elo_summary, support.expected_goals_summary,
                   support.event_summary, support.holdout_summary):
        with pytest.raises(ResearchFileMissing):
            loader(tmp_path)


def test_decisions_cover_every_deployment_question():
    assert list(DECISIONS.columns) == ["Research question", "Evidence", "Outcome", "In the app"]
    deployed = DECISIONS.set_index("Research question")["In the app"]
    assert deployed.str.startswith("Yes").sum() == 1  # only the three 1X2 models
    text = " ".join(DECISIONS["Research question"]).lower()
    for topic in ("elo", "poisson", "over/under", "adaptive"):
        assert topic in text


@pytest.mark.skipif(not HAS_RESEARCH, reason="experiment results not generated")
def test_research_summaries_match_the_saved_results():
    from src.backtest import load_backtest, pooled_metrics

    outcome = support.outcome_model_summary()
    expected = pooled_metrics(load_backtest().predictions)
    np.testing.assert_allclose(outcome["log_loss"], expected.loc[outcome.index, "log_loss"])
    bookmaker = [name for name in outcome.index if name.startswith("Bookmaker")][0]
    assert outcome.index[0] == bookmaker  # the bookmaker has the lowest pooled log loss
    assert outcome["seasons_best"].sum() == outcome["seasons"].iloc[0]  # one winner per season

    elo = support.elo_summary()
    assert list(elo.index) == ["Logistic Regression", "Random Forest", "XGBoost"]
    np.testing.assert_allclose(elo["difference"], elo["log_loss_with_elo"] - elo["log_loss"])
    assert (elo["seasons_improved"] <= elo["seasons"]).all()

    goals = support.expected_goals_summary()
    assert goals["deviance"].idxmin() != "League average"

    events, source = support.event_summary()
    assert source == "scoring_drift" and len(events) == 8
    assert events.loc["Event frequency", "Over 2.5: seasons better than static"] is pd.NA
    assert set(support.holdout_summary()["split"]) == {"validation", "test"}


@pytest.mark.skipif(not (OUTPUTS_DIR / "events" / "predictions.csv").exists(), reason="event results not generated")
def test_event_summary_falls_back_to_the_commit_16_results(tmp_path):
    (tmp_path / "events").mkdir()
    shutil.copy(OUTPUTS_DIR / "events" / "predictions.csv", tmp_path / "events" / "predictions.csv")
    table, source = support.event_summary(tmp_path)
    assert source == "events" and len(table) == 6
    assert "Adaptive event frequency" not in table.index


# ---------------------------------------------------------------- the app


def app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(PROJECT / "app.py"), default_timeout=180)


def all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [w.value for w in at.warning] + [i.value for i in at.info]
    return " ".join(parts)


def clear_app_caches():
    """Forget data and models cached by earlier app runs in this test process."""
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()


def test_app_explains_missing_match_data(monkeypatch):
    """A fresh clone has no data: the app must say how to create it instead of failing (runs without data)."""
    import src.preprocessing

    def missing(*args, **kwargs):
        raise FileNotFoundError("data/processed/matches.csv not found.")

    monkeypatch.setattr(src.preprocessing, "load_matches", missing)
    clear_app_caches()
    try:
        at = app().run()
    finally:
        clear_app_caches()
    assert not at.exception
    errors = [e for e in at.error if e.proto.title == "Match data not found"]
    assert len(errors) == 1
    assert "python -m src.data_loader" in errors[0].value and "python -m src.preprocessing" in errors[0].value
    assert not at.tabs  # nothing else is rendered


@pytest.mark.skipif(not (PROCESSED_DATA_DIR / "matches.csv").exists(), reason="processed data not generated")
def test_app_explains_missing_models(monkeypatch):
    import src.predict

    def missing(*args, **kwargs):
        raise src.predict.ModelFilesMissing("logistic_regression.joblib was not found. Run `python -m src.train`.")

    monkeypatch.setattr(src.predict, "load_saved_models", missing)
    clear_app_caches()
    try:
        at = app().run()
    finally:
        clear_app_caches()
    assert not at.exception
    errors = [e for e in at.error if e.proto.title == "Saved models not found"]
    assert len(errors) == 1 and "python -m src.train" in errors[0].value
    assert not at.tabs


@pytest.mark.skipif(not HAS_APP_INPUTS, reason="processed data or saved models not generated")
def test_app_has_three_sections_in_order():
    at = app().run()
    assert not at.exception
    labels = [tab.label.split(" ", 1)[1] for tab in at.tabs]
    assert labels == TABS
    headers = [h.value for h in at.header]
    assert headers.index("Goals & BTTS Analysis") < headers.index("Research & methodology")
    subheaders = [s.value for s in at.subheader]
    for title in ("Predictions by model", "Model comparison", "At a glance", "Key findings",
                  "1X2 models versus the bookmaker", "Elo ratings", "Goals models", "Decisions", "Limitations"):
        assert title in subheaders
    text = all_text(at)
    assert "Model output" in text
    assert "Descriptive history, not a forecast" in [i.proto.title for i in at.info]
    assert "not genuine" not in text  # upcoming mode: no training-data warning


@pytest.mark.skipif(not HAS_APP_INPUTS, reason="processed data or saved models not generated")
def test_app_explains_missing_research_files_without_breaking(monkeypatch, tmp_path):
    monkeypatch.setattr(support, "RESEARCH_ROOT", tmp_path)
    at = app().run()
    assert not at.exception
    missing = [i for i in at.info if i.proto.title == "Saved results not found"]
    assert len(missing) == 5  # 1X2 backtest, Elo, expected goals, events, saved-model table
    commands = " ".join(i.value for i in missing)
    for command in ("python -m src.backtest", "python -m src.elo_experiment", "python -m src.goals_backtest",
                    "python -m src.event_backtest", "python -m src.train"):
        assert command in commands
    assert len(at.dataframe) >= 3  # the predictions and goals tables still render
    assert "Decisions" in [s.value for s in at.subheader]  # the documented decisions need no files


@pytest.mark.skipif(not HAS_APP_INPUTS, reason="processed data or saved models not generated")
def test_app_warns_when_the_data_is_stale(monkeypatch):
    from src.predict import latest_completed_date
    from src.preprocessing import load_matches

    latest = latest_completed_date(load_matches()).date()
    monkeypatch.setattr(support, "today", lambda: latest + dt.timedelta(days=STALE_AFTER_DAYS + 20))
    at = app().run()
    assert not at.exception
    stale = [w for w in at.warning if w.proto.title == "Stale match data"]
    assert stale and f"{latest:%d %b %Y}" in stale[0].value and f"{STALE_AFTER_DAYS + 20} days ago" in stale[0].value

    monkeypatch.setattr(support, "today", lambda: latest + dt.timedelta(days=1))
    at = app().run()
    assert not [w for w in at.warning if w.proto.title == "Stale match data"]
    date_key = [d.key for d in at.date_input if d.key.startswith("date_")][0]
    at.date_input(key=date_key).set_value(latest + dt.timedelta(days=STALE_AFTER_DAYS + 30)).run()
    stale = [w.value for w in at.warning if w.proto.title == "Stale match data"]
    assert len(stale) == 1 and f"{STALE_AFTER_DAYS + 30} days after" in stale[0]


@pytest.mark.skipif(not HAS_APP_INPUTS, reason="processed data or saved models not generated")
def test_historical_demo_on_a_training_season_is_flagged():
    from src.preprocessing import load_matches

    matches = load_matches()
    fixture = matches[matches["season_start"] == 2010].sort_values("date").iloc[0]
    at = app().run()
    at.segmented_control(key="mode").set_value("Historical demo").run()
    at.selectbox(key="season").set_value(2010).run()
    at.selectbox(key="home").set_value(fixture["home_team"]).run()
    at.selectbox(key="away").set_value(fixture["away_team"]).run()
    at.date_input(key="date_Historical demo_2010").set_value(fixture["date"].date()).run()
    assert not at.exception
    assert any("not a genuine out-of-sample forecast" in w.value for w in at.warning)
    assert any(i.proto.title == "Actual result" for i in at.info)
    assert not [w for w in at.warning if w.proto.title == "Stale match data"]  # past fixtures are not stale

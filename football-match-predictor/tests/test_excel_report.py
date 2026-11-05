"""Tests for the Excel workbook (src.excel_report): sheets, fixtures, metrics and formulas."""

import re
from datetime import date

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

import src.excel_report as report
from src.bookmaker import BOOKMAKER_LABEL, bookmaker_probabilities
from src.evaluate import OUTCOMES, PROBABILITY_COLUMNS, evaluate
from src.excel_report import (
    BASELINE,
    COMP_HEADER,
    FORECASTS,
    MODELS,
    PRED_HEADER,
    SHEETS,
    build_workbook,
    collect_report_data,
    comparison_row,
    pred_cols,
)

TEAMS = ["Atalanta", "Bologna", "Como", "Genoa", "Inter", "Lecce"]  # 6 teams -> 30 matches a season
SEASONS = [2022, 2023, 2024, 2025, 2026]
TEST_PLAYED = 12  # the "current" season is incomplete
NO_ODDS_FIXTURE = 3  # one validation fixture without draw odds


def make_matches(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season in SEASONS:
        day = pd.Timestamp(f"{season}-08-20")
        pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
        if season == SEASONS[-1]:
            pairs = pairs[:TEST_PLAYED]
        for k, (home, away) in enumerate(pairs):
            probs = rng.dirichlet([4, 2.5, 3])
            result = OUTCOMES[rng.choice(3, p=probs)]
            hg, ag = {"H": (2, 0), "D": (1, 1), "A": (0, 1)}[result]
            odds = 1 / (probs * 1.055)
            rows.append({
                "season": f"{season}-{str(season + 1)[-2:]}", "season_start": season,
                "date": day + pd.Timedelta(days=k), "home_team": home, "away_team": away,
                "home_goals": hg, "away_goals": ag, "result": result,
                "b365_home": round(odds[0], 2), "b365_draw": round(odds[1], 2), "b365_away": round(odds[2], 2),
            })
    matches = pd.DataFrame(rows)
    matches["date"] = matches["date"].astype("datetime64[ns]")
    validation = matches.index[matches["season_start"] == 2025]
    matches.loc[validation[NO_ODDS_FIXTURE], "b365_draw"] = np.nan
    return matches


def make_predictions(matches, seed=1):
    """Saved results tables (as written by src.train) for every model and split."""
    rng = np.random.default_rng(seed)
    predictions = {}
    for split, season in (("validation", 2025), ("test", 2026)):
        rows = matches[matches["season_start"] == season]
        predictions[split] = {}
        for model in MODELS:
            probs = rng.dirichlet([3, 2, 2.5], size=len(rows))
            table = rows[["date", "home_team", "away_team"]].reset_index(drop=True)
            table["actual"] = rows["result"].to_numpy()
            table["predicted"] = [OUTCOMES[i] for i in probs.argmax(axis=1)]
            table[PROBABILITY_COLUMNS] = probs
            table["correct"] = table["actual"] == table["predicted"]
            # stored in a different order than the workbook uses
            predictions[split][model] = table.iloc[::-1].reset_index(drop=True)
    return predictions


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    matches = make_matches()
    predictions = make_predictions(matches)
    data = collect_report_data(matches, predictions, train_last_season=2024, validation_season=2025,
                               test_season=2026, n_resamples=200, generated=date(2026, 11, 5))
    path = tmp_path_factory.mktemp("excel") / "report.xlsx"
    build_workbook(data).save(path)
    return {"matches": matches, "predictions": predictions, "data": data, "wb": load_workbook(path)}


def table_rows(ws, header_row, n):
    return list(ws.iter_rows(min_row=header_row + 1, max_row=header_row + n, values_only=True))


# ---------------------------------------------------------------- structure


def test_workbook_has_the_expected_sheets_in_order(built):
    assert built["wb"].sheetnames == [
        "Overview", "Historical Trends", "Model Comparison", "Validation 2025-26", "Test 2026-27",
        "Model vs Bookmaker", "Match Data",
    ]


def test_excel_tables_filters_charts_and_formatting_are_present(built):
    wb = built["wb"]
    tables = {name for ws in wb.worksheets for name in ws.tables}
    assert {"MatchData", "SeasonTrends", "ModelComparison", "ValidationPredictions", "TestPredictions",
            "ModelVsBookmaker"} <= tables
    for name in ("Historical Trends", "Model Comparison", "Validation 2025-26", "Model vs Bookmaker"):
        assert len(wb[name].conditional_formatting) > 0, name
    assert len(wb["Overview"]._charts) == 3
    assert len(wb["Historical Trends"]._charts) == 2
    assert len(wb["Model Comparison"]._charts) == 2
    assert wb.calculation.fullCalcOnLoad  # Excel computes the formulas when the file opens


def test_formulas_only_use_functions_excel_and_libreoffice_both_support(built):
    unsupported = re.compile(r"\b(XLOOKUP|XMATCH|FILTER|SORT|UNIQUE|SEQUENCE|LET|LAMBDA)\(")
    formulas = [c.value for ws in built["wb"].worksheets for row in ws.iter_rows() for c in row
                if isinstance(c.value, str) and c.value.startswith("=")]
    assert len(formulas) > 1000
    assert not [f for f in formulas if unsupported.search(f)]


# ---------------------------------------------------------------- fixtures


@pytest.mark.parametrize("split, sheet", [("validation", "Validation 2025-26"), ("test", "Test 2026-27")])
def test_prediction_sheets_list_every_fixture_with_its_saved_probabilities(built, split, sheet):
    ws = built["wb"][sheet]
    saved = {m: t.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
             for m, t in built["predictions"][split].items()}
    n = len(saved[MODELS[0]])
    rows = table_rows(ws, PRED_HEADER, n)

    assert ws.cell(row=PRED_HEADER + n + 1, column=1).value is None  # nothing after the last fixture
    for i, row in enumerate(rows):
        expected = saved[MODELS[0]].iloc[i]
        assert (pd.Timestamp(row[0]), row[1], row[2], row[3]) == (
            expected["date"], expected["home_team"], expected["away_team"], expected["actual"])
        for k, model in enumerate(MODELS):
            c = pred_cols(k)
            got = [row[c[o] - 1] for o in OUTCOMES]
            np.testing.assert_allclose(got, saved[model].loc[i, PROBABILITY_COLUMNS].to_numpy(float), rtol=1e-12)


def test_per_match_formulas_point_at_their_own_row(built):
    ws = built["wb"]["Validation 2025-26"]
    r = PRED_HEADER + 5
    c = pred_cols(0)  # Logistic Regression: E, F, G
    assert ws.cell(row=r, column=c["pick"]).value == f'=IF(E{r}=MAX(E{r}:G{r}),"H",IF(F{r}=MAX(E{r}:G{r}),"D","A"))'
    assert ws.cell(row=r, column=c["log_loss"]).value == f'=-LN(IF($D{r}="H",E{r},IF($D{r}="D",F{r},G{r})))'
    assert ws.cell(row=r, column=c["brier"]).value == (
        f'=(E{r}-($D{r}="H"))^2+(F{r}-($D{r}="D"))^2+(G{r}-($D{r}="A"))^2')


def test_incomplete_season_is_clearly_labelled(built):
    wb = built["wb"]
    test_ws, val_ws = wb["Test 2026-27"], wb["Validation 2025-26"]
    assert f"{TEST_PLAYED} of 30 matches, season in progress" in test_ws["A1"].value
    assert "in progress" in test_ws["A4"].value.lower()
    assert "30 matches, complete season" in val_ws["A1"].value
    assert val_ws["A4"].value is None
    overview_text = " ".join(str(c.value) for row in wb["Overview"].iter_rows() for c in row if c.value)
    assert f"{TEST_PLAYED} of 30 matches, season in progress" in overview_text


# ---------------------------------------------------------------- metrics


def expected_metrics(built, split, season, last_train):
    matches = built["matches"]
    saved = {m: t.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
             for m, t in built["predictions"][split].items()}
    actual = saved[MODELS[0]]["actual"]
    results = {}
    for model, table in saved.items():
        probs = table[PROBABILITY_COLUMNS].set_axis(OUTCOMES, axis=1)
        results[model] = evaluate(actual, probs)
    season_rows = matches[matches["season_start"] == season].sort_values(["date", "home_team"])
    book = bookmaker_probabilities(season_rows).dropna()
    results[BOOKMAKER_LABEL] = evaluate(season_rows.loc[book.index, "result"], book)
    shares = matches.loc[matches["season_start"] <= last_train, "result"].value_counts(normalize=True)
    baseline = pd.DataFrame([shares.reindex(OUTCOMES).to_numpy()] * len(actual), columns=OUTCOMES)
    results[BASELINE] = evaluate(actual, baseline)
    return results


@pytest.mark.parametrize("k, split, season, last_train", [(0, "validation", 2025, 2024), (1, "test", 2026, 2025)])
def test_model_comparison_metrics_match_an_independent_evaluation(built, k, split, season, last_train):
    ws = built["wb"]["Model Comparison"]
    expected = expected_metrics(built, split, season, last_train)
    columns = {"matches": 5, "accuracy": 6, "macro_f1": 7, "log_loss": 8, "brier": 9}
    for forecast in FORECASTS:
        r = comparison_row(k, forecast)
        assert ws.cell(row=r, column=4).value == forecast
        assert ws.cell(row=r, column=1).value == split.capitalize()
        for key, c in columns.items():
            assert ws.cell(row=r, column=c).value == pytest.approx(expected[forecast][key]), (forecast, key)


def test_sample_sizes_and_bookmaker_coverage(built):
    ws = built["wb"]["Model Comparison"]
    assert ws.cell(row=comparison_row(0, MODELS[0]), column=5).value == 30
    assert ws.cell(row=comparison_row(0, BOOKMAKER_LABEL), column=5).value == 29  # one fixture without odds
    assert ws.cell(row=comparison_row(1, MODELS[0]), column=5).value == TEST_PLAYED
    assert ws.cell(row=COMP_HEADER + 1, column=15).value.startswith("=IF(AND(ABS(")  # Excel cross-check


def test_model_vs_bookmaker_uses_the_right_odds_for_each_fixture(built):
    ws = built["wb"]["Model vs Bookmaker"]
    matches = built["matches"].set_index(["date", "home_team", "away_team"])
    header = next(r for r in range(1, 80) if ws.cell(row=r, column=1).value == "Split")
    rows = [row for row in ws.iter_rows(min_row=header + 1, values_only=True) if row[0]]

    assert len(rows) == 29 + TEST_PLAYED
    assert {r[0] for r in rows} == {"Validation 2025-26", "Test 2026-27"}
    for row in rows:
        source = matches.loc[(pd.Timestamp(row[1]), row[2], row[3])]
        assert row[4] == source["result"]
        assert list(row[5:8]) == pytest.approx(source[["b365_home", "b365_draw", "b365_away"]].tolist())
    r = header + 1
    assert ws.cell(row=r, column=9).value == f"=1/F{r}+1/G{r}+1/H{r}"
    assert ws.cell(row=r, column=10).value == f"=(1/F{r})/$I{r}"


def test_historical_trends_has_one_formula_row_per_season(built):
    ws = built["wb"]["Historical Trends"]
    seasons = [ws.cell(row=r, column=1).value for r in range(6, 6 + len(SEASONS))]
    assert seasons == ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
    assert ws.cell(row=6, column=3).value == 6  # teams
    assert ws.cell(row=6, column=5).value.startswith("=COUNTIFS('Match Data'!")
    assert ws.cell(row=6, column=6).value == '=IF(E6<D6,"In progress","Complete")'


def test_match_data_holds_every_completed_match(built):
    ws = built["wb"]["Match Data"]
    n = len(built["matches"])
    assert ws.max_row == report.DATA_HEADER + n
    assert ws.cell(row=report.DATA_HEADER + n, column=12).value.startswith("=IF(COUNT(")


def test_findings_never_claim_significance(built):
    text = " ".join(report.findings(built["data"])).lower()
    assert "significant" not in text


# ---------------------------------------------------------------- validation of inputs


def test_predictions_from_another_season_are_rejected():
    matches = make_matches()
    predictions = make_predictions(matches)
    predictions["validation"], predictions["test"] = predictions["test"], predictions["validation"]
    with pytest.raises(ValueError, match="outside"):
        collect_report_data(matches, predictions, train_last_season=2024, validation_season=2025,
                            test_season=2026, n_resamples=50)


def test_write_report_and_cli(tmp_path, monkeypatch):
    matches = make_matches()
    data = collect_report_data(matches, make_predictions(matches), train_last_season=2024,
                               validation_season=2025, test_season=2026, n_resamples=50)
    path = report.write_report(tmp_path / "sub" / "book.xlsx", data=data)
    assert path.exists()

    monkeypatch.setattr(report, "write_report", lambda output: output)
    report.main(["--output", str(tmp_path / "cli.xlsx")])


# ---------------------------------------------------------------- real outputs (if generated)


def test_real_workbook_matches_saved_predictions(tmp_path):
    try:
        predictions = report.load_saved_predictions()
        matches = report.load_matches()
    except FileNotFoundError:
        pytest.skip("Saved predictions or processed data not found; run python -m src.train first")
    data = collect_report_data(matches, predictions, n_resamples=50)
    path = tmp_path / "real.xlsx"
    build_workbook(data).save(path)
    wb = load_workbook(path)

    for k, (split, season_sheet) in enumerate((("validation", data.splits["validation"].sheet),
                                               ("test", data.splits["test"].sheet))):
        saved = predictions[split][MODELS[0]].sort_values(["date", "home_team", "away_team"])
        rows = table_rows(wb[season_sheet], PRED_HEADER, len(saved))
        assert [(r[1], r[2], r[3]) for r in rows] == list(
            saved[["home_team", "away_team", "actual"]].itertuples(index=False, name=None))
        probs = saved[PROBABILITY_COLUMNS].set_axis(OUTCOMES, axis=1).reset_index(drop=True)
        expected = evaluate(saved["actual"].reset_index(drop=True), probs)
        ws = wb["Model Comparison"]
        r = comparison_row(k, MODELS[0])
        assert ws.cell(row=r, column=8).value == pytest.approx(expected["log_loss"])
        assert ws.cell(row=r, column=5).value == len(saved)

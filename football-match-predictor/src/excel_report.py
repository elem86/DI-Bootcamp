"""Excel workbook with the project's data analysis and model results.

Usage (from the project folder, after ``python -m src.train``)::

    python -m src.excel_report

writes ``outputs/excel/serie_a_prediction_analysis.xlsx`` (git-ignored).

Nothing is retrained: the workbook is built from the processed match data
and the per-match predictions saved by ``src.train``. Python supplies the
data (matches, odds, predicted probabilities) and the reference metrics from
``src.evaluate``. Excel formulas recompute everything that can be derived
from those inputs: per-match picks, log loss and Brier score, bookmaker
probabilities, season statistics and summary metrics. A check column on the
Model Comparison sheet confirms that the formulas reproduce the Python metrics.

Sheets
------
Overview, Historical Trends, Model Comparison, Validation <season>,
Test <season>, Model vs Bookmaker, Match Data (the source table).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.chart.label import DataLabelList
from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule, DataBarRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.properties import CalcProperties
from openpyxl.worksheet.table import Table, TableStyleInfo

from src.bookmaker import BOOKMAKER_LABEL, FIXTURE_KEYS, ODDS_COLUMNS, Benchmark, build_benchmark
from src.config import EXCEL_DIR
from src.evaluate import OUTCOMES, bootstrap_table, evaluate, metrics_table, probabilities_from_results
from src.models import MODEL_SPECS
from src.preprocessing import load_matches
from src.train import TEST_SEASON, TRAIN_LAST_SEASON, VALIDATION_SEASON, load_predictions, season_name

EXCEL_FILE = EXCEL_DIR / "serie_a_prediction_analysis.xlsx"

BASELINE = "Baseline"
MODELS = [spec.label for spec in MODEL_SPECS.values()]
FORECASTS = MODELS + [BOOKMAKER_LABEL, BASELINE]
SHORT = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB", BOOKMAKER_LABEL: "Bookmaker",
         BASELINE: "Baseline"}
BOOTSTRAP_PAIRS = [(m, BOOKMAKER_LABEL) for m in MODELS] + [
    (MODELS[0], MODELS[1]), (MODELS[0], MODELS[2]), (MODELS[1], MODELS[2])]

SHEETS = {
    "overview": "Overview",
    "history": "Historical Trends",
    "comparison": "Model Comparison",
    "bookmaker": "Model vs Bookmaker",
    "data": "Match Data",
}


# ====================================================================== data


@dataclass
class SplitData:
    """One evaluation season: its fixtures, every forecast and the baseline."""

    name: str  # "validation" or "test"
    season_start: int
    first_train_season: int
    last_train_season: int
    fixtures: pd.DataFrame  # date, home_team, away_team, actual (all predicted fixtures)
    model_probs: dict[str, pd.DataFrame]  # model label -> H/D/A, aligned with fixtures
    benchmark: Benchmark  # fixtures with complete odds, incl. bookmaker probabilities
    baseline: pd.Series  # training outcome shares (H, D, A)
    scheduled: int  # matches in the full season
    metrics: pd.DataFrame  # one row per forecast (src.evaluate.metrics_table)
    bootstrap: pd.DataFrame  # paired bootstrap of log-loss differences

    @property
    def season(self) -> str:
        return season_name(self.season_start)

    @property
    def sheet(self) -> str:
        return f"{self.name.capitalize()} {self.season}"

    @property
    def played(self) -> int:
        return len(self.fixtures)

    @property
    def complete(self) -> bool:
        return self.played >= self.scheduled

    @property
    def status(self) -> str:
        return "Complete" if self.complete else "In progress"

    @property
    def description(self) -> str:
        if self.complete:
            return f"{self.played} matches, complete season"
        return f"{self.played} of {self.scheduled} matches, season in progress"


@dataclass
class ReportData:
    matches: pd.DataFrame  # completed matches, in date order
    splits: dict[str, SplitData]
    generated: date


def load_saved_predictions() -> dict[str, dict[str, pd.DataFrame]]:
    """Saved per-match predictions of every model: ``{split: {model label: table}}``."""
    return {
        split: {spec.label: load_predictions(name, split) for name, spec in MODEL_SPECS.items()}
        for split in ("validation", "test")
    }


def teams_in_season(matches: pd.DataFrame, season_start: int) -> int:
    """Clubs that appear in the season's matches."""
    season = matches[matches["season_start"] == season_start]
    return len(pd.unique(season[["home_team", "away_team"]].to_numpy().ravel()))


def scheduled_matches(matches: pd.DataFrame, season_start: int) -> int:
    """Matches in a full double round-robin season: teams x (teams - 1)."""
    teams = teams_in_season(matches, season_start)
    return teams * (teams - 1)


def collect_report_data(
    matches: pd.DataFrame,
    predictions: dict[str, dict[str, pd.DataFrame]],
    train_last_season: int = TRAIN_LAST_SEASON,
    validation_season: int = VALIDATION_SEASON,
    test_season: int = TEST_SEASON,
    n_resamples: int = 10_000,
    generated: date | None = None,
) -> ReportData:
    """Everything the workbook shows, from the match data and saved predictions."""
    played = matches[matches["result"].notna()]
    played = played.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    first_season = int(played["season_start"].min())

    plan = {"validation": (validation_season, train_last_season), "test": (test_season, validation_season)}
    splits = {}
    for name, (season, last_train) in plan.items():
        saved = {label: predictions[name][label] for label in MODELS}
        # Checks that every model covers the same fixtures and that results agree
        benchmark = build_benchmark(saved, played)
        ordered = {
            label: frame.sort_values(FIXTURE_KEYS, kind="stable").reset_index(drop=True)
            for label, frame in saved.items()
        }
        fixtures = ordered[MODELS[0]][FIXTURE_KEYS + ["actual"]].copy()
        fixtures["date"] = fixtures["date"].astype("datetime64[ns]")
        in_season = fixtures.merge(
            played.loc[played["season_start"] == season, FIXTURE_KEYS], on=FIXTURE_KEYS, how="left", indicator=True
        )
        if (in_season["_merge"] != "both").any():
            raise ValueError(f"{name} predictions contain fixtures outside the {season_name(season)} season")

        model_probs = {label: probabilities_from_results(frame) for label, frame in ordered.items()}
        train = played[played["season_start"] <= last_train]
        baseline = train["result"].astype(str).value_counts(normalize=True).reindex(OUTCOMES, fill_value=0.0)
        actual = fixtures["actual"]
        baseline_probs = pd.DataFrame([baseline.to_numpy()] * len(fixtures), columns=OUTCOMES)

        results = {label: evaluate(actual, probs) for label, probs in model_probs.items()}
        results[BOOKMAKER_LABEL] = evaluate(benchmark.actual, benchmark.probabilities[BOOKMAKER_LABEL])
        results[BASELINE] = evaluate(actual, baseline_probs)

        splits[name] = SplitData(
            name=name,
            season_start=season,
            first_train_season=first_season,
            last_train_season=last_train,
            fixtures=fixtures,
            model_probs=model_probs,
            benchmark=benchmark,
            baseline=baseline,
            scheduled=scheduled_matches(played, season),
            metrics=metrics_table(results).loc[FORECASTS],
            bootstrap=bootstrap_table(
                benchmark.actual, benchmark.probabilities, BOOTSTRAP_PAIRS, n_resamples=n_resamples
            ),
        )
    return ReportData(matches=played, splits=splits, generated=generated or date.today())


# ====================================================================== styling

FONT = "Arial"
INK, INK_2, MUTED = "0B0B0B", "52514E", "898781"
INPUT_BLUE = "0000FF"  # values supplied by the Python pipeline
LINK_GREEN = "008000"  # formulas that pull values from another sheet
HEADER_FILL = PatternFill("solid", fgColor="104281")
GROUP_FILL = PatternFill("solid", fgColor="DCE9F9")
TILE_FILL = PatternFill("solid", fgColor="F0EFEC")
NOTE_FILL = PatternFill("solid", fgColor="FFF4D6")
GOOD_FILL = PatternFill("solid", fgColor="D9F2E6")
BAD_FILL = PatternFill("solid", fgColor="FBE0D6")
THIN = Side(style="thin", color="C3C2B7")
COLORS = {"validation": "2A78D6", "test": "EB6834", "H": "2A78D6", "D": "1BAF7A", "A": "EB6834"}

PCT, PCT0, DEC3, DEC2 = "0.0%", "0%", "0.000", "0.00"
SIGNED = "+0.000;-0.000;0.000"
DATE = "dd mmm yyyy"


def font(color=INK, bold=False, size=10, italic=False):
    return Font(name=FONT, color=color, bold=bold, size=size, italic=italic)


def put(ws, row, col, value, fmt=None, color=INK, bold=False, size=10, italic=False, fill=None, wrap=False,
        align=None):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = font(color, bold, size, italic)
    if fmt:
        cell.number_format = fmt
    if fill:
        cell.fill = fill
    if wrap or align:
        cell.alignment = Alignment(wrap_text=wrap, horizontal=align, vertical="top" if wrap else None)
    return cell


def title(ws, text, subtitle=None, note=None):
    put(ws, 1, 1, text, bold=True, size=16)
    if subtitle:
        put(ws, 2, 1, subtitle, color=INK_2, size=11)
    if note:
        put(ws, 3, 1, note, color=MUTED, italic=True, size=9)


def header_row(ws, row, col, names, fill=HEADER_FILL):
    for offset, name in enumerate(names):
        cell = put(ws, row, col + offset, name, color="FFFFFF", bold=True, fill=fill, wrap=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")


def group_row(ws, row, groups):
    """Labels spanning several columns above a table, e.g. ``[(5, 11, "Logistic Regression")]``."""
    for first, last, label in groups:
        ws.merge_cells(start_row=row, start_column=first, end_row=row, end_column=last)
        cell = put(ws, row, first, label, bold=True, fill=GROUP_FILL, align="center")
        for col in range(first, last + 1):
            ws.cell(row=row, column=col).border = Border(bottom=THIN)
            ws.cell(row=row, column=col).fill = GROUP_FILL


def add_table(ws, name, first_row, last_row, first_col, last_col, style="TableStyleLight9"):
    ref = f"{get_column_letter(first_col)}{first_row}:{get_column_letter(last_col)}{last_row}"
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(name=style, showRowStripes=True)
    ws.add_table(table)
    return table


def widths(ws, values: dict[str, float]):
    for column, width in values.items():
        ws.column_dimensions[column].width = width


def col(n: int) -> str:
    return get_column_letter(n)


def sheet_ref(sheet: str) -> str:
    return f"'{sheet}'" if " " in sheet or "-" in sheet else sheet


def style_chart(chart, title_text, y_title=None, y_format=None, height=7.5, width=16):
    chart.title = title_text
    chart.title.overlay = False  # keep the title and legend outside the plot area
    chart.height, chart.width = height, width
    chart.legend.position = "b"
    chart.legend.overlay = False
    chart.x_axis.delete = False  # openpyxl hides axes in new Excel versions otherwise
    chart.y_axis.delete = False
    if y_title:
        chart.y_axis.title = y_title
    if y_format:
        chart.y_axis.number_format = y_format
    return chart


def color_series(series, color, line=False):
    if line:
        series.graphicalProperties.line.solidFill = color
        series.graphicalProperties.line.width = 22000
        series.marker.symbol = "circle"
        series.marker.size = 5
        series.marker.graphicalProperties.solidFill = color
        series.marker.graphicalProperties.line.solidFill = color
        series.smooth = False
    else:
        series.graphicalProperties.solidFill = color
        series.graphicalProperties.line.solidFill = color


# ====================================================================== layout
# Fixed positions that other sheets refer to.

DATA_HEADER = 4  # Match Data table header row
PRED_SUMMARY_FIRST = 6  # prediction sheets: summary rows (one per model)
PRED_HEADER = 12  # prediction sheets: table header row (group labels one row above)
PRED_BLOCK = 7  # columns per model: P(H), P(D), P(A), pick, correct, log loss, Brier
PRED_FIRST_MODEL_COL = 5
COMP_HEADER = 7  # Model Comparison table header row
BOOK_SUMMARY_HEADER = 6  # Model vs Bookmaker summary header row


def pred_cols(model_index: int) -> dict[str, int]:
    start = PRED_FIRST_MODEL_COL + model_index * PRED_BLOCK
    return {"H": start, "D": start + 1, "A": start + 2, "pick": start + 3, "correct": start + 4,
            "log_loss": start + 5, "brier": start + 6}


def pred_rows(split: SplitData) -> tuple[int, int]:
    return PRED_HEADER + 1, PRED_HEADER + split.played


def comparison_row(split_index: int, forecast: str) -> int:
    return COMP_HEADER + 1 + split_index * len(FORECASTS) + FORECASTS.index(forecast)


def baseline_header_row() -> int:
    return comparison_row(1, FORECASTS[-1]) + 4


# per-match formulas shared by the prediction and bookmaker sheets
def pick_formula(h, d, a):
    return f'=IF({h}=MAX({h}:{a}),"H",IF({d}=MAX({h}:{a}),"D","A"))'


def log_loss_formula(result, h, d, a):
    return f'=-LN(IF({result}="H",{h},IF({result}="D",{d},{a})))'


def brier_formula(result, h, d, a):
    return f'=({h}-({result}="H"))^2+({d}-({result}="D"))^2+({a}-({result}="A"))^2'


# ====================================================================== sheets


def write_match_data(wb, data: ReportData):
    ws = wb.create_sheet(SHEETS["data"])
    ws.sheet_properties.tabColor = "898781"
    title(ws, "Match data: every completed Serie A match", f"{len(data.matches):,} matches from Football-Data.co.uk "
          "(processed by src/preprocessing.py)", "Blue = data from the Python pipeline; black = Excel formula. "
          "Bet365 odds are blank where the data has none (before 2002-03 and a few later matches).")
    columns = ["Season", "Season start", "Date", "Home team", "Away team", "Home goals", "Away goals", "Result",
               "B365 home", "B365 draw", "B365 away", "Overround"]
    header_row(ws, DATA_HEADER, 1, columns)
    m = data.matches
    odds = [ODDS_COLUMNS[o] for o in OUTCOMES]
    for i, row in enumerate(m.itertuples(index=False), start=DATA_HEADER + 1):
        values = [row.season, row.season_start, row.date.to_pydatetime(), row.home_team, row.away_team,
                  int(row.home_goals), int(row.away_goals), row.result]
        values += [None if pd.isna(getattr(row, c)) else float(getattr(row, c)) for c in odds]
        for j, value in enumerate(values, start=1):
            put(ws, i, j, value, fmt=DATE if j == 3 else (DEC2 if j >= 9 else None), color=INPUT_BLUE)
        put(ws, i, 12, f'=IF(COUNT(I{i}:K{i})=3,1/I{i}+1/J{i}+1/K{i},"")', fmt="0.0000")
    last = DATA_HEADER + len(m)
    add_table(ws, "MatchData", DATA_HEADER, last, 1, len(columns))
    ws.freeze_panes = ws.cell(row=DATA_HEADER + 1, column=1)
    widths(ws, {"A": 10, "B": 9, "C": 13, "D": 16, "E": 16, "F": 8, "G": 8, "H": 8, "I": 9, "J": 9, "K": 9,
                "L": 11})
    ws.cell(row=DATA_HEADER, column=12).comment = Comment(
        "Sum of the three implied probabilities 1/odds. 1.055 means a 5.5% bookmaker margin.", "src.excel_report")
    return last


def write_history(wb, data: ReportData, data_last_row: int):
    ws = wb.create_sheet(SHEETS["history"])
    ws.sheet_properties.tabColor = "1BAF7A"
    seasons = data.matches[["season", "season_start"]].drop_duplicates().sort_values("season_start")
    title(ws, "Historical Serie A statistics and outcome trends",
          f"{len(seasons)} seasons, {seasons['season'].iloc[0]} to {seasons['season'].iloc[-1]}; every value "
          "below is an Excel formula over the Match Data sheet",
          "The latest season is still in progress (Status column), so its shares are based on fewer matches.")
    header = 5
    columns = ["Season", "Season start", "Teams", "Scheduled matches", "Matches played", "Status", "Home wins",
               "Draws", "Away wins", "Home win %", "Draw %", "Away win %", "Goals per match",
               "Home goals per match", "Away goals per match", "Matches with odds", "Odds coverage",
               "Average bookmaker margin"]
    header_row(ws, header, 1, columns)
    d = sheet_ref(SHEETS["data"])
    r0, r1 = DATA_HEADER + 1, data_last_row
    rng = {key: f"{d}!${letter}${r0}:${letter}${r1}" for key, letter in
           {"season": "A", "start": "B", "hg": "F", "ag": "G", "result": "H", "bh": "I", "bd": "J", "ba": "K",
            "over": "L"}.items()}
    for i, s in enumerate(seasons.itertuples(index=False), start=header + 1):
        put(ws, i, 1, s.season, color=INPUT_BLUE)
        put(ws, i, 2, int(s.season_start), color=INPUT_BLUE)
        put(ws, i, 3, teams_in_season(data.matches, s.season_start), color=INPUT_BLUE)
        put(ws, i, 4, f"=C{i}*(C{i}-1)")
        put(ws, i, 5, f"=COUNTIFS({rng['season']},A{i})")
        put(ws, i, 6, f'=IF(E{i}<D{i},"In progress","Complete")')
        for c, outcome in zip((7, 8, 9), OUTCOMES):
            put(ws, i, c, f'=COUNTIFS({rng["season"]},$A{i},{rng["result"]},"{outcome}")')
        for c, source in zip((10, 11, 12), ("G", "H", "I")):
            put(ws, i, c, f"={source}{i}/$E{i}", fmt=PCT)
        put(ws, i, 13, f"=(SUMIFS({rng['hg']},{rng['season']},A{i})+SUMIFS({rng['ag']},{rng['season']},A{i}))/E{i}",
            fmt=DEC2)
        put(ws, i, 14, f"=SUMIFS({rng['hg']},{rng['season']},A{i})/E{i}", fmt=DEC2)
        put(ws, i, 15, f"=SUMIFS({rng['ag']},{rng['season']},A{i})/E{i}", fmt=DEC2)
        put(ws, i, 16, f'=COUNTIFS({rng["season"]},A{i},{rng["bh"]},"<>",{rng["bd"]},"<>",{rng["ba"]},"<>")')
        put(ws, i, 17, f"=P{i}/E{i}", fmt=PCT0)
        put(ws, i, 18, f'=IFERROR(AVERAGEIFS({rng["over"]},{rng["season"]},A{i})-1,"")', fmt=PCT)
    last = header + len(seasons)
    add_table(ws, "SeasonTrends", header, last, 1, len(columns))
    ws.cell(row=header, column=3).comment = Comment("Clubs that appear in the season's matches.", "src.excel_report")

    total = last + 2
    put(ws, total, 1, "All seasons", bold=True)
    put(ws, total, 5, f"=SUM(E{header + 1}:E{last})", bold=True)
    for c in (7, 8, 9, 16):
        put(ws, total, c, f"=SUM({col(c)}{header + 1}:{col(c)}{last})", bold=True)
    for c, source in zip((10, 11, 12), ("G", "H", "I")):
        put(ws, total, c, f"={source}{total}/$E{total}", fmt=PCT, bold=True)
    put(ws, total, 13, f"=SUMPRODUCT(M{header + 1}:M{last},E{header + 1}:E{last})/E{total}", fmt=DEC2, bold=True)
    put(ws, total, 14, f"=SUMPRODUCT(N{header + 1}:N{last},E{header + 1}:E{last})/E{total}", fmt=DEC2, bold=True)
    put(ws, total, 15, f"=SUMPRODUCT(O{header + 1}:O{last},E{header + 1}:E{last})/E{total}", fmt=DEC2, bold=True)
    put(ws, total, 17, f"=P{total}/E{total}", fmt=PCT0, bold=True)

    ws.conditional_formatting.add(
        f"A{header + 1}:R{last}",
        FormulaRule(formula=[f'$F{header + 1}="In progress"'], fill=NOTE_FILL, font=Font(name=FONT, italic=True)))
    ws.conditional_formatting.add(f"J{header + 1}:J{last}", ColorScaleRule(
        start_type="min", start_color="F0EFEC", end_type="max", end_color="2A78D6"))
    ws.conditional_formatting.add(f"M{header + 1}:M{last}", DataBarRule(
        start_type="num", start_value=0, end_type="max", color="86B6EF"))

    shares = LineChart()
    style_chart(shares, "Share of home wins, draws and away wins", y_format="0%", width=15)
    for c, outcome in zip((10, 11, 12), OUTCOMES):
        series = Series(Reference(ws, min_col=c, min_row=header, max_row=last), title_from_data=True)
        color_series(series, COLORS[outcome], line=True)
        shares.series.append(series)
    shares.set_categories(Reference(ws, min_col=1, min_row=header + 1, max_row=last))
    ws.add_chart(shares, f"A{total + 3}")

    goals = LineChart()
    style_chart(goals, "Goals per match", y_format="0.0", width=15)
    for c, color in zip((13, 14, 15), ("0B0B0B", COLORS["H"], COLORS["A"])):
        series = Series(Reference(ws, min_col=c, min_row=header, max_row=last), title_from_data=True)
        color_series(series, color, line=True)
        goals.series.append(series)
    goals.set_categories(Reference(ws, min_col=1, min_row=header + 1, max_row=last))
    ws.add_chart(goals, f"J{total + 3}")

    ws.freeze_panes = ws.cell(row=header + 1, column=2)
    widths(ws, {"A": 10, "B": 9, "C": 7, "D": 10, "E": 9, "F": 11, "G": 8, "H": 8, "I": 8, "J": 9, "K": 9,
                "L": 9, "M": 9, "N": 10, "O": 10, "P": 9, "Q": 9, "R": 11})
    ws.row_dimensions[header].height = 42
    return {"header": header, "last": last}


def write_predictions(wb, split: SplitData):
    ws = wb.create_sheet(split.sheet)
    ws.sheet_properties.tabColor = COLORS[split.name]
    role = ("used to choose each model's settings (feature set and hyperparameters)" if split.name == "validation"
            else "never used for any modelling choice")
    trained = f"{season_name(split.first_train_season)} to {season_name(split.last_train_season)}"
    title(ws, f"{split.name.capitalize()} predictions, {split.season}: {split.description}",
          f"Models trained on {trained}; this season was {role}.",
          "Blue = saved model probabilities (outputs/predictions); black = Excel formulas. Pick = most likely "
          "outcome; log loss = -LN(probability of the actual result); Brier = squared error over H, D, A.")
    if not split.complete:
        put(ws, 4, 1, f"Season in progress: {split.played} of {split.scheduled} matches played. Results will "
                      "change as the season continues.", bold=True, fill=NOTE_FILL)

    first, last = pred_rows(split)
    summary_cols = ["Forecast", "Matches", "Accuracy", "Log loss", "Brier score", "Predicted draws",
                    "Draws correct"]
    header_row(ws, PRED_SUMMARY_FIRST - 1, 1, summary_cols)
    for k, model in enumerate(MODELS):
        r, c = PRED_SUMMARY_FIRST + k, pred_cols(k)
        pick, corr = col(c["pick"]), col(c["correct"])
        put(ws, r, 1, model, bold=True)
        put(ws, r, 2, f"=COUNTA(D{first}:D{last})")
        put(ws, r, 3, f"=AVERAGE({corr}{first}:{corr}{last})", fmt=PCT)
        put(ws, r, 4, f"=AVERAGE({col(c['log_loss'])}{first}:{col(c['log_loss'])}{last})", fmt=DEC3)
        put(ws, r, 5, f"=AVERAGE({col(c['brier'])}{first}:{col(c['brier'])}{last})", fmt=DEC3)
        put(ws, r, 6, f'=COUNTIF({pick}{first}:{pick}{last},"D")')
        put(ws, r, 7, f'=COUNTIFS({pick}{first}:{pick}{last},"D",D{first}:D{last},"D")')

    groups, columns = [], ["Date", "Home team", "Away team", "Result"]
    for k, model in enumerate(MODELS):
        c, s = pred_cols(k), SHORT[model]
        groups.append((c["H"], c["brier"], model))
        columns += [f"{s} P(H)", f"{s} P(D)", f"{s} P(A)", f"{s} pick", f"{s} correct", f"{s} log loss",
                    f"{s} Brier"]
    columns.append("All models agree")
    agree_col = len(columns)
    group_row(ws, PRED_HEADER - 1, groups)
    header_row(ws, PRED_HEADER, 1, columns)

    for i, fixture in enumerate(split.fixtures.itertuples(index=False)):
        r = first + i
        put(ws, r, 1, fixture.date.to_pydatetime(), fmt=DATE, color=INPUT_BLUE)
        put(ws, r, 2, fixture.home_team, color=INPUT_BLUE)
        put(ws, r, 3, fixture.away_team, color=INPUT_BLUE)
        put(ws, r, 4, fixture.actual, color=INPUT_BLUE, align="center")
        for k, model in enumerate(MODELS):
            c = pred_cols(k)
            probs = split.model_probs[model].iloc[i]
            for outcome in OUTCOMES:
                put(ws, r, c[outcome], float(probs[outcome]), fmt=PCT, color=INPUT_BLUE)
            h, d, a = (f"{col(c[o])}{r}" for o in OUTCOMES)
            put(ws, r, c["pick"], pick_formula(h, d, a), align="center")
            put(ws, r, c["correct"], f"=IF({col(c['pick'])}{r}=$D{r},1,0)", align="center")
            put(ws, r, c["log_loss"], log_loss_formula(f"$D{r}", h, d, a), fmt=DEC3)
            put(ws, r, c["brier"], brier_formula(f"$D{r}", h, d, a), fmt=DEC3)
        picks = [f"{col(pred_cols(k)['pick'])}{r}" for k in range(len(MODELS))]
        put(ws, r, agree_col, f'=IF(AND({picks[0]}={picks[1]},{picks[1]}={picks[2]}),"Yes","No")', align="center")

    add_table(ws, f"{split.name.capitalize()}Predictions", PRED_HEADER, last, 1, agree_col)
    for k in range(len(MODELS)):
        c = pred_cols(k)
        ws.conditional_formatting.add(
            f"{col(c['H'])}{first}:{col(c['A'])}{last}",
            ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF", end_type="num", end_value=1,
                           end_color="2A78D6"))
        correct = f"{col(c['correct'])}{first}:{col(c['correct'])}{last}"
        ws.conditional_formatting.add(correct, CellIsRule(operator="equal", formula=["1"], fill=GOOD_FILL))
        ws.conditional_formatting.add(correct, CellIsRule(operator="equal", formula=["0"], fill=BAD_FILL))
        ws.conditional_formatting.add(
            f"{col(c['log_loss'])}{first}:{col(c['log_loss'])}{last}",
            ColorScaleRule(start_type="min", start_color="FFFFFF", end_type="max", end_color="EB6834"))
    ws.freeze_panes = ws.cell(row=first, column=5)
    widths(ws, {"A": 19, "B": 15, "C": 15, "D": 8} | {col(c): 8 for c in range(5, agree_col)} |
           {col(agree_col): 10})
    ws.row_dimensions[PRED_HEADER].height = 30
    ws.row_dimensions[PRED_SUMMARY_FIRST - 1].height = 30
    return {"first": first, "last": last}


def write_bookmaker(wb, data: ReportData):
    ws = wb.create_sheet(SHEETS["bookmaker"])
    ws.sheet_properties.tabColor = "EDA100"
    splits = list(data.splits.values())
    title(ws, "Models vs bookmaker: probabilities on the same fixtures",
          "Bet365 1X2 odds turned into probabilities (1/odds, divided by the overround to remove the margin). "
          "The odds are a benchmark only, never a model feature.",
          "Each fixture is matched on date, home and away team, and its result must agree with the match data. "
          "Fixtures without all three odds are left out. No betting-profit analysis.")

    # ---- per-match table (written first so the summary can refer to its rows)
    split_rows = sum(len(s.benchmark.fixtures) for s in splits)
    boot_rows = sum(len(s.bootstrap) for s in splits)
    boot_header = BOOK_SUMMARY_HEADER + len(splits) + 4
    header = boot_header + boot_rows + 5
    first, last = header + 1, header + split_rows
    columns = ["Split", "Date", "Home team", "Away team", "Result", "B365 home", "B365 draw", "B365 away",
               "Overround", "Bookmaker P(H)", "Bookmaker P(D)", "Bookmaker P(A)", "Bookmaker pick"]
    model_prob_col = {}
    for model in MODELS:
        model_prob_col[model] = len(columns) + 1
        columns += [f"{SHORT[model]} P(H)", f"{SHORT[model]} P(D)", f"{SHORT[model]} P(A)"]
    loss_col = {}
    for forecast in [BOOKMAKER_LABEL] + MODELS:
        loss_col[forecast] = len(columns) + 1
        columns.append(f"{SHORT[forecast]} log loss")
    columns += ["Bookmaker Brier", "Lowest log loss", "LR - bookmaker P(H)"]
    brier_col, best_col, gap_col = len(columns) - 2, len(columns) - 1, len(columns)
    group_row(ws, header - 1, [
        (6, 9, "Bet365 decimal odds"), (10, 13, "Bookmaker, margin removed"),
        (model_prob_col[MODELS[0]], model_prob_col[MODELS[-1]] + 2, "Model probabilities"),
        (loss_col[BOOKMAKER_LABEL], brier_col, "Per-match scores (lower is better)"),
    ])
    header_row(ws, header, 1, columns)

    lookup = data.matches.set_index(FIXTURE_KEYS)
    r = first
    split_ranges = {}
    for split in splits:
        bench = split.benchmark
        start = r
        for i, fixture in enumerate(bench.fixtures.itertuples(index=False)):
            key = (pd.Timestamp(fixture.date), fixture.home_team, fixture.away_team)
            odds = lookup.loc[key, [ODDS_COLUMNS[o] for o in OUTCOMES]]
            put(ws, r, 1, split.sheet, color=INPUT_BLUE)
            put(ws, r, 2, pd.Timestamp(fixture.date).to_pydatetime(), fmt=DATE, color=INPUT_BLUE)
            put(ws, r, 3, fixture.home_team, color=INPUT_BLUE)
            put(ws, r, 4, fixture.away_team, color=INPUT_BLUE)
            put(ws, r, 5, fixture.actual, color=INPUT_BLUE, align="center")
            for c, value in zip((6, 7, 8), odds):
                put(ws, r, c, float(value), fmt=DEC2, color=INPUT_BLUE)
            put(ws, r, 9, f"=1/F{r}+1/G{r}+1/H{r}", fmt="0.0000")
            for c, odds_col in zip((10, 11, 12), ("F", "G", "H")):
                put(ws, r, c, f"=(1/{odds_col}{r})/$I{r}", fmt=PCT)
            put(ws, r, 13, pick_formula(f"J{r}", f"K{r}", f"L{r}"), align="center")
            for model in MODELS:
                probs = bench.probabilities[model].iloc[i]
                for offset, outcome in enumerate(OUTCOMES):
                    put(ws, r, model_prob_col[model] + offset, float(probs[outcome]), fmt=PCT, color=INPUT_BLUE)
            for forecast in [BOOKMAKER_LABEL] + MODELS:
                c0 = 10 if forecast == BOOKMAKER_LABEL else model_prob_col[forecast]
                h, d, a = (f"{col(c0 + k)}{r}" for k in range(3))
                put(ws, r, loss_col[forecast], log_loss_formula(f"$E{r}", h, d, a), fmt=DEC3)
            put(ws, r, brier_col, brier_formula(f"$E{r}", f"J{r}", f"K{r}", f"L{r}"), fmt=DEC3)
            losses = f"{col(loss_col[BOOKMAKER_LABEL])}{r}:{col(loss_col[MODELS[-1]])}{r}"
            names = ",".join(f'"{SHORT[f]}"' for f in [BOOKMAKER_LABEL] + MODELS)
            put(ws, r, best_col, f"=CHOOSE(MATCH(MIN({losses}),{losses},0),{names})", align="center")
            put(ws, r, gap_col, f"={col(model_prob_col[MODELS[0]])}{r}-J{r}", fmt="+0.0%;-0.0%;0.0%")
            r += 1
        split_ranges[split.name] = (start, r - 1)
    add_table(ws, "ModelVsBookmaker", header, last, 1, len(columns))
    ws.conditional_formatting.add(f"J{first}:L{last}", ColorScaleRule(
        start_type="num", start_value=0, start_color="FFFFFF", end_type="num", end_value=1, end_color="EDA100"))
    model_first, model_last = model_prob_col[MODELS[0]], model_prob_col[MODELS[-1]] + 2
    ws.conditional_formatting.add(f"{col(model_first)}{first}:{col(model_last)}{last}", ColorScaleRule(
        start_type="num", start_value=0, start_color="FFFFFF", end_type="num", end_value=1, end_color="2A78D6"))
    ws.conditional_formatting.add(f"{col(gap_col)}{first}:{col(gap_col)}{last}", ColorScaleRule(
        start_type="num", start_value=-0.3, start_color="EB6834", mid_type="num", mid_value=0,
        mid_color="FFFFFF", end_type="num", end_value=0.3, end_color="2A78D6"))

    # ---- summary (formulas over the per-match table)
    put(ws, BOOK_SUMMARY_HEADER - 1, 1, "Summary by season (Excel formulas over the per-match table below)",
        bold=True, size=11)
    summary_cols = ["Season", "Status", "Fixtures with odds", "Odds coverage", "Average overround",
                    "Average margin", "Bookmaker accuracy", "Bookmaker log loss", "Bookmaker Brier",
                    "LR log loss", "RF log loss", "XGB log loss", "Lowest average log loss"]
    header_row(ws, BOOK_SUMMARY_HEADER, 1, summary_cols)
    split_col = f"$A${first}:$A${last}"
    summary_rows = {}
    for k, split in enumerate(splits):
        rr = BOOK_SUMMARY_HEADER + 1 + k
        summary_rows[split.name] = rr
        put(ws, rr, 1, split.sheet, bold=True)
        put(ws, rr, 2, f"{split.status} ({split.played} of {split.scheduled})", color=INPUT_BLUE)
        put(ws, rr, 3, f"=COUNTIF({split_col},A{rr})")
        put(ws, rr, 4, f"=C{rr}/{split.played}", fmt=PCT0)
        ws.cell(row=rr, column=4).comment = Comment(
            f"Fixtures with complete odds / {split.played} predicted fixtures.", "src.excel_report")
        put(ws, rr, 5, f"=AVERAGEIFS($I${first}:$I${last},{split_col},A{rr})", fmt="0.0000")
        put(ws, rr, 6, f"=E{rr}-1", fmt=PCT)
        put(ws, rr, 7, f"=SUMPRODUCT(({split_col}=A{rr})*($M${first}:$M${last}=$E${first}:$E${last}))/C{rr}",
            fmt=PCT)
        for c, forecast in zip((8, 10, 11, 12), [BOOKMAKER_LABEL] + MODELS):
            loss = col(loss_col[forecast])
            put(ws, rr, c, f"=AVERAGEIFS(${loss}${first}:${loss}${last},{split_col},A{rr})", fmt=DEC3)
        put(ws, rr, 9, f"=AVERAGEIFS(${col(brier_col)}${first}:${col(brier_col)}${last},{split_col},A{rr})",
            fmt=DEC3)
        put(ws, rr, 13, f'=IF(H{rr}<=MIN(J{rr}:L{rr}),"Bookmaker",'
                        f'CHOOSE(MATCH(MIN(J{rr}:L{rr}),J{rr}:L{rr},0),"LR","RF","XGB"))', align="center")

    # ---- bootstrap
    put(ws, boot_header - 1, 1, "Is a difference larger than chance? Paired bootstrap of log loss (A - B), "
        "10,000 resamples of the matches, 95% interval", bold=True, size=11)
    boot_cols = ["Season", "Forecast A", "Forecast B", "Matches", "Difference (A - B)", "95% CI low",
                 "95% CI high", "Interval includes 0", "Reading"]
    header_row(ws, boot_header, 1, boot_cols)
    rr = boot_header + 1
    for split in splits:
        for row in split.bootstrap.itertuples(index=False):
            put(ws, rr, 1, split.sheet)
            put(ws, rr, 2, SHORT[row.a], color=INPUT_BLUE)
            put(ws, rr, 3, SHORT[row.b], color=INPUT_BLUE)
            put(ws, rr, 4, int(row.matches), color=INPUT_BLUE)
            put(ws, rr, 5, float(row.difference), fmt=SIGNED, color=INPUT_BLUE)
            put(ws, rr, 6, float(row.ci_low), fmt=SIGNED, color=INPUT_BLUE)
            put(ws, rr, 7, float(row.ci_high), fmt=SIGNED, color=INPUT_BLUE)
            put(ws, rr, 8, f'=IF(AND(F{rr}<=0,G{rr}>=0),"Yes","No")', align="center")
            put(ws, rr, 9, f'=IF(H{rr}="Yes","Cannot tell apart",IF(E{rr}<0,B{rr},C{rr})&" lower")')
            rr += 1
    boot_last = rr - 1
    ws.conditional_formatting.add(f"H{boot_header + 1}:H{boot_last}",
                                  CellIsRule(operator="equal", formula=['"No"'], fill=NOTE_FILL))
    put(ws, boot_last + 1, 1, "Computed in Python (src.evaluate.paired_bootstrap). Assumes independent matches, "
        "covers one season each and makes several comparisons without correction: an interval that only just "
        "excludes 0 is a hint, not proof.", color=MUTED, italic=True, size=9)

    # ---- scatter: LR vs bookmaker home-win probability (validation rows)
    v_first, v_last = split_ranges["validation"]
    scatter = ScatterChart()
    style_chart(scatter, f"Home-win probability, {data.splits['validation'].sheet}", y_format="0%", height=9,
                width=12)
    scatter.x_axis.title = "Bookmaker P(H)"
    scatter.y_axis.title = "LR P(H)"
    scatter.x_axis.number_format = "0%"
    points = Series(Reference(ws, min_col=model_prob_col[MODELS[0]], min_row=v_first, max_row=v_last),
                    Reference(ws, min_col=10, min_row=v_first, max_row=v_last), title="Matches")
    points.marker.symbol = "circle"
    points.marker.size = 4
    points.marker.graphicalProperties.solidFill = COLORS["validation"]
    points.marker.graphicalProperties.line.solidFill = COLORS["validation"]
    points.graphicalProperties.line.noFill = True
    scatter.series.append(points)
    scatter.x_axis.scaling.min = scatter.y_axis.scaling.min = 0
    scatter.x_axis.scaling.max = scatter.y_axis.scaling.max = 1
    scatter.legend = None
    ws.add_chart(scatter, f"L{boot_header - 1}")

    ws.freeze_panes = ws.cell(row=first, column=6)
    widths(ws, {"A": 18, "B": 21, "C": 15, "D": 15, "E": 13} | {col(c): 10 for c in range(6, len(columns) + 1)})
    for row in (header, BOOK_SUMMARY_HEADER, boot_header):
        ws.row_dimensions[row].height = 45
    return {"summary_rows": summary_rows, "first": first, "last": last, "split_ranges": split_ranges}


def write_comparison(wb, data: ReportData, pred_layout, book_layout, data_last_row):
    ws = wb.create_sheet(SHEETS["comparison"], 1)
    ws.sheet_properties.tabColor = "104281"
    splits = list(data.splits.values())
    title(ws, "Model comparison: validation and test results",
          " | ".join(f"{s.sheet}: {s.description}" for s in splits),
          "Accuracy and macro F1 measure the top pick; log loss and Brier score measure the probabilities "
          "(lower is better). Blue values come from src.evaluate; the Check column recomputes them in Excel.")
    put(ws, 4, 1, "The validation season was used to choose each model's settings, so its scores are slightly "
        "optimistic. The test season was never used for any choice"
        + ("" if splits[-1].complete else " and is still in progress") + ".", fill=NOTE_FILL)
    put(ws, 5, 1, "Bookmaker rows use only fixtures with complete Bet365 odds (Matches column).", color=MUTED,
        italic=True, size=9)

    columns = ["Split", "Season", "Season status", "Forecast", "Matches", "Accuracy", "Macro F1", "Log loss",
               "Brier score", "Home-win accuracy", "Draw accuracy", "Away-win accuracy", "Log-loss rank",
               "Log-loss gain vs baseline", "Check (recomputed in Excel)"]
    header_row(ws, COMP_HEADER, 1, columns)
    first = COMP_HEADER + 1
    last = comparison_row(len(splits) - 1, FORECASTS[-1])
    hist = sheet_ref(SHEETS["history"])
    base_header = baseline_header_row()
    for k, split in enumerate(splits):
        pred = sheet_ref(split.sheet)
        for forecast in FORECASTS:
            r = comparison_row(k, forecast)
            m = split.metrics.loc[forecast]
            put(ws, r, 1, split.name.capitalize(), bold=True)
            put(ws, r, 2, split.season)
            put(ws, r, 3, f'=INDEX({hist}!$F:$F,MATCH({split.season_start},{hist}!$B:$B,0))', color=LINK_GREEN)
            put(ws, r, 4, forecast, bold=forecast == BOOKMAKER_LABEL)
            put(ws, r, 5, int(m["matches"]), color=INPUT_BLUE)
            for c, key, fmt in ((6, "accuracy", PCT), (7, "macro_f1", DEC3), (8, "log_loss", DEC3),
                                (9, "brier", DEC3), (10, "accuracy_H", PCT), (11, "accuracy_D", PCT),
                                (12, "accuracy_A", PCT)):
                put(ws, r, c, float(m[key]), fmt=fmt, color=INPUT_BLUE)
            block = f"$H${comparison_row(k, FORECASTS[0])}:$H${comparison_row(k, FORECASTS[-1])}"
            put(ws, r, 13, f"=COUNTIF({block},\"<\"&H{r})+1", align="center")
            put(ws, r, 14, f"=$H${comparison_row(k, BASELINE)}-H{r}", fmt=SIGNED)
            if forecast in MODELS:
                s = PRED_SUMMARY_FIRST + MODELS.index(forecast)
                refs = (f"{pred}!$C${s}", f"{pred}!$D${s}", f"{pred}!$E${s}")
            elif forecast == BOOKMAKER_LABEL:
                s = book_layout["summary_rows"][split.name]
                book = sheet_ref(SHEETS["bookmaker"])
                refs = (f"{book}!$G${s}", f"{book}!$H${s}", f"{book}!$I${s}")
            else:
                s = base_header + 1 + k
                refs = (f"$G${s}", f"$H${s}", f"$I${s}")
            put(ws, r, 15, f'=IF(AND(ABS(F{r}-{refs[0]})<0.000001,ABS(H{r}-{refs[1]})<0.000001,'
                           f'ABS(I{r}-{refs[2]})<0.000001),"OK","Differs")', align="center")
    add_table(ws, "ModelComparison", COMP_HEADER, last, 1, len(columns), style="TableStyleMedium2")
    ws.conditional_formatting.add(f"H{first}:H{last}", ColorScaleRule(
        start_type="min", start_color="D9F2E6", end_type="max", end_color="FBE0D6"))
    ws.conditional_formatting.add(f"F{first}:F{last}", DataBarRule(
        start_type="num", start_value=0, end_type="num", end_value=1, color="86B6EF"))
    ws.conditional_formatting.add(f"M{first}:M{last}", CellIsRule(operator="equal", formula=["1"],
                                                                  fill=GOOD_FILL, font=Font(name=FONT, bold=True)))
    ws.conditional_formatting.add(f"O{first}:O{last}", CellIsRule(operator="notEqual", formula=['"OK"'],
                                                                  fill=BAD_FILL))

    # ---- baseline probabilities, recomputed from the Match Data sheet
    put(ws, base_header - 1, 1, "Baseline: always predicts the most frequent training outcome, with the training "
        "outcome shares as its probabilities (Excel formulas over Match Data)", bold=True, size=11)
    header_row(ws, base_header, 1, ["Split", "First training season", "Last training season", "P(H)", "P(D)",
                                    "P(A)", "Accuracy", "Log loss", "Brier score"])
    d = sheet_ref(SHEETS["data"])
    starts = f"{d}!$B${DATA_HEADER + 1}:$B${data_last_row}"
    results = f"{d}!$H${DATA_HEADER + 1}:$H${data_last_row}"
    for k, split in enumerate(splits):
        r = base_header + 1 + k
        first_p, last_p = pred_layout[split.name]["first"], pred_layout[split.name]["last"]
        res = f"{sheet_ref(split.sheet)}!$D${first_p}:$D${last_p}"
        put(ws, r, 1, split.name.capitalize(), bold=True)
        put(ws, r, 2, split.first_train_season, color=INPUT_BLUE)
        put(ws, r, 3, split.last_train_season, color=INPUT_BLUE)
        for c, outcome in zip((4, 5, 6), OUTCOMES):
            put(ws, r, c, f'=COUNTIFS({starts},"<="&$C{r},{results},"{outcome}")/COUNTIFS({starts},"<="&$C{r})',
                fmt=PCT)
        put(ws, r, 7, f'=COUNTIF({res},CHOOSE(MATCH(MAX(D{r}:F{r}),D{r}:F{r},0),"H","D","A"))/COUNTA({res})',
            fmt=PCT)
        put(ws, r, 8, f'=-(COUNTIF({res},"H")*LN(D{r})+COUNTIF({res},"D")*LN(E{r})+COUNTIF({res},"A")*LN(F{r}))'
                      f"/COUNTA({res})", fmt=DEC3)
        put(ws, r, 9, f'=SUMSQ(D{r}:F{r})+1-2*(COUNTIF({res},"H")*D{r}+COUNTIF({res},"D")*E{r}'
                      f'+COUNTIF({res},"A")*F{r})/COUNTA({res})', fmt=DEC3)

    # ---- chart data and charts
    chart_header = base_header + len(splits) + 3
    put(ws, chart_header - 1, 1, "Chart data (linked to the table above)", bold=True, size=11)
    header_row(ws, chart_header, 1, ["Forecast"] + [f"{s.sheet} accuracy" for s in splits]
               + [f"{s.sheet} log-loss gain" for s in splits])
    for j, forecast in enumerate(FORECASTS):
        r = chart_header + 1 + j
        put(ws, r, 1, SHORT[forecast])
        for k in range(len(splits)):
            put(ws, r, 2 + k, f"=F{comparison_row(k, forecast)}", fmt=PCT)
            put(ws, r, 2 + len(splits) + k, f"=N{comparison_row(k, forecast)}", fmt=SIGNED)
    chart_last = chart_header + len(FORECASTS)
    for chart, anchor in comparison_charts(ws, chart_header, chart_last, splits):
        ws.add_chart(chart, anchor)

    ws.freeze_panes = ws.cell(row=first, column=5)
    widths(ws, {"A": 20, "B": 10, "C": 12, "D": 22, "E": 9, "F": 10, "G": 9, "H": 9, "I": 9, "J": 10, "K": 10,
                "L": 10, "M": 9, "N": 11, "O": 13})
    for row in (COMP_HEADER, base_header, chart_header):
        ws.row_dimensions[row].height = 45
    return {"chart_header": chart_header, "chart_last": chart_last}


def comparison_charts(ws, chart_header, chart_last, splits):
    """Accuracy and log-loss gain over the baseline, validation vs test."""
    charts = []
    for offset, (name, fmt, y_title) in enumerate((
        ("Accuracy by forecast", "0%", None),
        ("Log-loss improvement over the baseline (higher is better)", "0.00", None),
    )):
        chart = BarChart()
        chart.type, chart.grouping = "col", "clustered"
        style_chart(chart, name, y_title=y_title, y_format=fmt)
        for k, split in enumerate(splits):
            column = 2 + offset * len(splits) + k
            series = Series(Reference(ws, min_col=column, min_row=chart_header, max_row=chart_last),
                            title_from_data=True)
            color_series(series, COLORS[split.name])
            chart.series.append(series)
        chart.set_categories(Reference(ws, min_col=1, min_row=chart_header + 1, max_row=chart_last))
        chart.y_axis.scaling.min = 0
        chart.dataLabels = DataLabelList(showVal=True, showSerName=False, showCatName=False,
                                         showLegendKey=False, showPercent=False)
        charts.append((chart, f"Q{COMP_HEADER + 18 * offset}"))
    return charts


def findings(data: ReportData) -> list[str]:
    """Key findings, written from the current numbers."""
    val, test = data.splits["validation"], data.splits["test"]
    vm, tm = val.metrics, test.metrics
    best_val = vm.loc[MODELS, "log_loss"].idxmin()
    lines = [
        f"All three models beat the baseline: validation log loss {vm.loc[MODELS, 'log_loss'].min():.3f}-"
        f"{vm.loc[MODELS, 'log_loss'].max():.3f} vs {vm.loc[BASELINE, 'log_loss']:.3f}, test accuracy "
        f"{tm.loc[MODELS, 'accuracy'].min():.0%}-{tm.loc[MODELS, 'accuracy'].max():.0%} vs "
        f"{tm.loc[BASELINE, 'accuracy']:.0%}.",
    ]
    model_pairs = [p for p in BOOTSTRAP_PAIRS if BOOKMAKER_LABEL not in p]
    all_overlap = all(
        bool(row.ci_includes_zero)
        for split in (val, test)
        for row in split.bootstrap.itertuples(index=False)
        if (row.a, row.b) in model_pairs
    )
    lines.append(
        "No model clearly beats another: every model-vs-model bootstrap interval includes 0 on both seasons."
        if all_overlap else
        "Some model-vs-model bootstrap intervals exclude 0; see the Model vs Bookmaker sheet before drawing "
        "conclusions.")
    book_val = vm.loc[BOOKMAKER_LABEL, "log_loss"]
    vs_book = val.bootstrap[val.bootstrap["b"] == BOOKMAKER_LABEL]
    if book_val < vm.loc[MODELS, "log_loss"].min():
        strength = ("the bootstrap intervals exclude 0, narrowly" if (~vs_book["ci_includes_zero"]).all()
                    else "but some bootstrap intervals include 0")
        lines.append(f"The bookmaker's probabilities are better on validation: log loss {book_val:.3f} vs "
                     f"{vm.loc[best_val, 'log_loss']:.3f} for the best model ({best_val}); {strength}.")
    else:
        lines.append(f"On validation the best model ({best_val}, {vm.loc[best_val, 'log_loss']:.3f}) matches or "
                     f"beats the bookmaker ({book_val:.3f}).")
    test_vs_book = test.bootstrap[test.bootstrap["b"] == BOOKMAKER_LABEL]
    lines.append(
        f"Test {test.season} ({test.description}): models log loss {tm.loc[MODELS, 'log_loss'].min():.3f}-"
        f"{tm.loc[MODELS, 'log_loss'].max():.3f} vs bookmaker {tm.loc[BOOKMAKER_LABEL, 'log_loss']:.3f}; "
        + ("every model-vs-bookmaker interval includes 0, so the sample cannot separate them."
           if test_vs_book["ci_includes_zero"].all() else "some intervals exclude 0; the sample is still small."))
    lines.append(
        f"Accuracy is not probability quality: compare the Accuracy and Log loss columns. On test the bookmaker's "
        f"accuracy is {tm.loc[BOOKMAKER_LABEL, 'accuracy']:.0%} with log loss "
        f"{tm.loc[BOOKMAKER_LABEL, 'log_loss']:.3f}.")
    lines.append(
        f"Draws are hard for everyone: validation draw accuracy is "
        f"{vm.loc[MODELS + [BOOKMAKER_LABEL], 'accuracy_D'].max():.0%} at best.")
    return lines


def write_overview(wb, data: ReportData, history, comparison):
    ws = wb.active
    ws.title = SHEETS["overview"]
    ws.sheet_properties.tabColor = "0B0B0B"
    ws.sheet_view.showGridLines = False
    val, test = data.splits["validation"], data.splits["test"]
    last_date = data.matches["date"].max()
    title(ws, "Serie A Match Predictor: analysis dashboard",
          f"Home win / draw / away win predictions from historical results. Data up to {last_date:%d %b %Y}; "
          f"workbook generated {data.generated:%d %b %Y}.",
          "Generated by python -m src.excel_report from saved predictions; no model was retrained. "
          "Blue = values from Python, black = Excel formulas, green = formulas linking another sheet.")

    comp = sheet_ref(SHEETS["comparison"])
    hist = sheet_ref(SHEETS["history"])
    vr = {f: comparison_row(0, f) for f in FORECASTS}
    tr = {f: comparison_row(1, f) for f in FORECASTS}
    vm = f"{comp}!$H${vr[MODELS[0]]}:$H${vr[MODELS[-1]]}"
    tacc = f"{comp}!$F${tr[MODELS[0]]}:$F${tr[MODELS[-1]]}"
    names_v = f"{comp}!$D${vr[MODELS[0]]}:$D${vr[MODELS[-1]]}"
    names_t = f"{comp}!$D${tr[MODELS[0]]}:$D${tr[MODELS[-1]]}"
    tiles = [
        ("Best model, validation log loss", f"=MIN({vm})", DEC3,
         f'=INDEX({names_v},MATCH(MIN({vm}),{vm},0))&", {val.played} matches"'),
        ("Bookmaker, validation log loss", f"={comp}!$H${vr[BOOKMAKER_LABEL]}", DEC3,
         '="Bet365, margin removed"'),
        ("Best model, test accuracy", f"=MAX({tacc})", PCT,
         f'=INDEX({names_t},MATCH(MAX({tacc}),{tacc},0))&", {test.played} matches"'),
        ("Test season progress",
         f'=INDEX({hist}!$E:$E,MATCH({test.season_start},{hist}!$B:$B,0))&" of "&'
         f'INDEX({hist}!$D:$D,MATCH({test.season_start},{hist}!$B:$B,0))', None,
         f'=INDEX({hist}!$F:$F,MATCH({test.season_start},{hist}!$B:$B,0))&", matches played"'),
    ]
    for k, (label, formula, fmt, caption) in enumerate(tiles):
        c = 1 + 3 * k
        for rr in range(5, 9):
            for cc in range(c, c + 2):
                ws.cell(row=rr, column=cc).fill = TILE_FILL
        put(ws, 5, c, label, color=INK_2, bold=True, fill=TILE_FILL)
        put(ws, 6, c, formula, fmt=fmt, color=LINK_GREEN, bold=True, size=20, fill=TILE_FILL, align="left")
        put(ws, 8, c, caption, color=INK_2, size=9, fill=TILE_FILL)
        ws.merge_cells(start_row=6, start_column=c, end_row=7, end_column=c + 1)

    put(ws, 10, 1, "Evaluation plan", bold=True, size=12)
    first_train = season_name(val.first_train_season)
    plan = [
        ("Data", f"Football-Data.co.uk Serie A results, {len(data.matches):,} completed matches "
                 f"({season_name(int(data.matches['season_start'].min()))} to {test.season})"),
        ("Training", f"{first_train} to {season_name(val.last_train_season)} for validation; "
                     f"{first_train} to {season_name(test.last_train_season)} for the final test models"),
        ("Validation", f"{val.season}: {val.description}. Used to choose each model's settings."),
        ("Test", f"{test.season}: {test.description}. Never used for any choice."),
        ("Models", "Logistic Regression, Random Forest, XGBoost; baseline = training outcome shares"),
        ("Benchmark", "Bet365 1X2 odds, margin removed; benchmark only, never a model feature"),
    ]
    for k, (label, text) in enumerate(plan):
        put(ws, 11 + k, 1, label, bold=True)
        put(ws, 11 + k, 2, text, color=INK_2)
        if label == "Test" and not test.complete:
            ws.cell(row=11 + k, column=2).fill = NOTE_FILL

    row = 11 + len(plan) + 1
    put(ws, row, 1, "Key findings", bold=True, size=12)
    lines = findings(data)
    for k, line in enumerate(lines):
        put(ws, row + 1 + k, 1, f"{k + 1}.", color=INK_2)
        put(ws, row + 1 + k, 2, line)
    row += len(lines) + 2
    put(ws, row, 1, "Findings are written from the numbers at generation time; uncertainty comes from a paired "
        "bootstrap (Model vs Bookmaker sheet). Small differences are not statistically established.",
        color=MUTED, italic=True, size=9)

    # Charts (about 16 rows tall), then the metric glossary
    charts_row = row + 2
    comparison_ws = wb[SHEETS["comparison"]]
    for k, (chart, _) in enumerate(comparison_charts(comparison_ws, comparison["chart_header"],
                                                     comparison["chart_last"], list(data.splits.values()))):
        chart.width = 12.5
        ws.add_chart(chart, f"{'A' if k == 0 else 'G'}{charts_row}")
    history_ws = wb[SHEETS["history"]]
    shares = LineChart()
    style_chart(shares, "Outcome shares by season", y_format="0%", width=12.5)
    for c, outcome in zip((10, 11, 12), OUTCOMES):
        series = Series(Reference(history_ws, min_col=c, min_row=history["header"], max_row=history["last"]),
                        title_from_data=True)
        color_series(series, COLORS[outcome], line=True)
        shares.series.append(series)
    shares.set_categories(Reference(history_ws, min_col=1, min_row=history["header"] + 1, max_row=history["last"]))
    ws.add_chart(shares, f"M{charts_row}")

    glossary_row = charts_row + 17
    put(ws, glossary_row, 1, "Metrics", bold=True, size=12)
    glossary = [
        ("Accuracy", "Share of matches where the most likely outcome happened (higher is better)."),
        ("Macro F1", "Average F1 score over home win, draw and away win; rewards getting draws right too."),
        ("Log loss", "-LN(probability given to the actual result), averaged; punishes confident mistakes "
                     "(lower is better)."),
        ("Brier score", "Squared error of the three probabilities against the result; 0 = perfect, "
                        "0.667 = one third each, 2 = certain and wrong."),
        ("Overround", "Sum of the bookmaker's implied probabilities 1/odds; the excess over 1 is its margin."),
    ]
    for k, (term, text) in enumerate(glossary):
        put(ws, glossary_row + 1 + k, 1, term, bold=True)
        put(ws, glossary_row + 1 + k, 2, text, color=INK_2)

    widths(ws, {"A": 16, "B": 14, "C": 4, "D": 16, "E": 14, "F": 4, "G": 16, "H": 14, "I": 4, "J": 16, "K": 14})


# ====================================================================== workbook


def build_workbook(data: ReportData) -> Workbook:
    wb = Workbook()
    # Build order: data first so later sheets can refer to its row numbers; tabs are reordered at the end.
    wb.active.title = SHEETS["overview"]
    data_last_row = write_match_data(wb, data)
    history = write_history(wb, data, data_last_row)
    pred_layout = {name: write_predictions(wb, split) for name, split in data.splits.items()}
    book_layout = write_bookmaker(wb, data)
    comparison = write_comparison(wb, data, pred_layout, book_layout, data_last_row)
    write_overview(wb, data, history, comparison)

    order = [SHEETS["overview"], SHEETS["history"], SHEETS["comparison"]] + \
            [split.sheet for split in data.splits.values()] + [SHEETS["bookmaker"], SHEETS["data"]]
    wb._sheets = [wb[name] for name in order]
    wb.active = 0
    for ws in wb.worksheets:
        ws.sheet_view.zoomScale = 90
    # openpyxl writes formulas without results: ask Excel to calculate everything on open
    wb.calculation = CalcProperties(fullCalcOnLoad=True)
    wb.properties.title = "Serie A match prediction analysis"
    wb.properties.creator = "football-match-predictor (src.excel_report)"
    return wb


def write_report(path: Path = EXCEL_FILE, data: ReportData | None = None) -> Path:
    data = data or collect_report_data(load_matches(), load_saved_predictions())
    path.parent.mkdir(parents=True, exist_ok=True)
    build_workbook(data).save(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the Excel analysis workbook from saved predictions.")
    parser.add_argument("--output", type=Path, default=EXCEL_FILE, help="Where to write the .xlsx file.")
    args = parser.parse_args(argv)
    path = write_report(args.output)
    try:
        shown = path.resolve().relative_to(Path.cwd())
    except ValueError:
        shown = path
    print(f"Saved workbook: {shown}")


if __name__ == "__main__":
    main()

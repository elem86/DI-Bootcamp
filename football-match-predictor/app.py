"""Serie A Match Predictor: an interactive Streamlit app.

Run from the project folder (after generating the data and models, see the README)::

    streamlit run app.py

The app never retrains anything and never reruns an experiment. It reads the
processed match data and the models saved by ``python -m src.train``, builds
the 50 pre-match features of the chosen fixture from matches completed
strictly before its date (``src/predict.py``), and shows each model's
probabilities. Three tabs keep three kinds of content apart:

* **Match predictions**: the saved 1X2 models' probabilities (model output).
* **Goals & BTTS history**: descriptive frequencies from past matches
  (``src/goal_stats.py``), not forecasts.
* **Research & methodology**: summaries of the completed experiments, read
  from the saved result files in ``outputs/`` (``src/app_support.py``). Those
  files are optional; when they are missing the app explains how to create them.
"""

import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

import src.app_support as support
from src.evaluate import OUTCOME_NAMES, OUTCOMES
from src.feature_engineering import SIDES, TEAM_FEATURES
from src.goal_stats import AVERAGES, EVENTS, GOAL_LINES, HISTORY_WINDOWS, fixture_goal_stats, line_key, rate, \
    team_goal_rows
from src.predict import (
    FixtureError,
    ModelFilesMissing,
    fixture_features,
    history_summary,
    latest_completed_date,
    load_saved_models,
    most_likely,
    outcome_table,
    played_fixture,
    predict_fixture,
    season_windows,
    simple_average,
)
from src.preprocessing import load_matches
from src.train import season_name

st.set_page_config(page_title="Serie A Match Predictor", page_icon=":material/sports_soccer:", layout="wide")

UPCOMING, DEMO = "Upcoming fixture", "Historical demo"
AVERAGE = "Simple average (unvalidated)"
# Home is blue and away orange everywhere in the app (outcomes and clubs); the models use three other hues.
HOME_COLOR, AWAY_COLOR, DRAW_COLOR = "#2a78d6", "#eb6834", "#c3c2b7"
MODEL_COLORS = {"Logistic Regression": "#1baf7a", "Random Forest": "#4a3aa7", "XGBoost": "#c44e9b",
                AVERAGE: "#52514e"}
OUTCOME_ORDER = [OUTCOME_NAMES[o] for o in OUTCOMES]
OUTCOME_COLORS = dict(zip(OUTCOME_ORDER, [HOME_COLOR, DRAW_COLOR, AWAY_COLOR]))
INK_2, SURFACE = "#52514e", "#fcfcfb"


# ---------------------------------------------------------------- cached loading


@st.cache_data(show_spinner="Loading match data...")
def load_data() -> pd.DataFrame:
    return load_matches()


@st.cache_resource(show_spinner="Loading saved models...")
def load_models():
    return load_saved_models()


@st.cache_data(show_spinner="Preparing goal statistics...")
def load_goal_rows() -> pd.DataFrame:
    """One row per club per completed match (src.goal_stats); filtered per fixture."""
    return team_goal_rows(load_data())


@st.cache_data(max_entries=256, show_spinner=False)
def cached_fixture_features(home: str, away: str, date: dt.date) -> pd.DataFrame:
    return fixture_features(load_data(), home, away, date)


RESEARCH_LOADERS = {
    "outcome": support.outcome_model_summary,
    "elo": support.elo_summary,
    "expected_goals": support.expected_goals_summary,
    "events": support.event_summary,
    "holdout": support.holdout_summary,
}


@st.cache_data(max_entries=32, show_spinner="Loading saved research results...")
def load_research(name: str, root: str):
    """A research summary from the saved result files (never reruns an experiment)."""
    return RESEARCH_LOADERS[name](root)


def next_saturday(start: dt.date) -> dt.date:
    return start + dt.timedelta(days=(5 - start.weekday()) % 7)


def keep_valid(key: str, options, fallback):
    """Reset a widget's stored value when it is no longer one of its options."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = fallback


# ---------------------------------------------------------------- data and models

st.title("Serie A Match Predictor")
st.caption("Home win, draw and away win probabilities from three machine-learning models trained on Serie A "
           "results, alongside descriptive goals statistics and the evidence behind what the app does and does not "
           "predict.")

try:
    matches = load_data()
except FileNotFoundError as error:
    st.error(f"{error}\n\nA fresh clone contains no data or models. From the project folder, run these four "
             "commands once, then restart the app (see *Getting started* in the README):\n\n"
             "```\npython -m src.data_loader\npython -m src.preprocessing\npython -m src.feature_engineering\n"
             "python -m src.train\n```", title="Match data not found")
    st.stop()

try:
    models = load_models()
except ModelFilesMissing as error:
    st.error(str(error), title="Saved models not found")
    st.stop()

windows = season_windows(matches)
latest_season = windows[max(windows)]
latest_date = latest_completed_date(matches).date()
today = support.today()
training = next(iter(models.values())).training_seasons
training_label = f"{season_name(training[0])} to {season_name(training[1])}" if training else "unknown"

# ---------------------------------------------------------------- controls

with st.sidebar:
    st.header("Fixture", divider="gray")
    mode = st.segmented_control("Mode", [UPCOMING, DEMO], default=UPCOMING, required=True, key="mode",
                                help="Upcoming fixture: any two clubs on a date after the latest data. Historical "
                                     "demo rebuilds any past fixture using only the results known before that day.")

    if mode == UPCOMING:
        window = latest_season
        min_date = latest_date + dt.timedelta(days=1)
        max_date = window.last_date.date()
        default_date = min(next_saturday(max(today, min_date)), max_date)
    else:
        seasons = sorted(windows, reverse=True)
        keep_valid("season", seasons, seasons[0])
        season = st.selectbox("Season", seasons, format_func=season_name, key="season")
        window = windows[season]
        min_date = window.first_date.date()
        max_date = min(window.last_date.date(), latest_date)
        default_date = max_date

    if min_date > max_date:
        st.error(f"The {window.label} season has no dates left after the latest data ({latest_date:%d %b %Y}).",
                 title="No upcoming dates")
        st.stop()

    teams = list(window.teams)
    keep_valid("home", teams, teams[0])
    home = st.selectbox("Home team", teams, key="home")
    away_options = [team for team in teams if team != home]
    keep_valid("away", away_options, away_options[0])
    away = st.selectbox("Away team", away_options, key="away",
                        help="The home team is not offered: a club cannot play itself.")
    date = st.date_input("Match date", value=default_date, min_value=min_date, max_value=max_date,
                         format="DD/MM/YYYY", key=f"date_{mode}_{window.season_start}")

    st.subheader("Data", divider="gray")
    freshness = support.data_freshness(latest_date, date, today)
    st.metric("Latest completed match in the data", f"{latest_date:%d %b %Y}",
              delta=f"{freshness.data_age_days} days ago" if freshness.data_age_days > 0 else "today",
              delta_color="inverse" if freshness.data_is_stale else "off", delta_arrow="off")
    st.caption(f"{len(matches):,} completed Serie A matches from {season_name(min(windows))} to "
               f"{latest_season.label}. Clubs listed for {window.label} come from that season's matches.")
    if mode == UPCOMING:
        st.caption(f"Future dates can be chosen from {min_date:%d %b %Y} to the end of {window.label} "
                   f"({max_date:%d %b %Y}). The app does not check the official Serie A calendar.")

# ---------------------------------------------------------------- features and predictions

try:
    features = cached_fixture_features(home, away, date)
    probabilities = predict_fixture(models, features)
except FixtureError as error:
    st.error(str(error), title="This fixture can't be predicted")
    st.stop()

history = history_summary(matches, date)

st.header(f"{home} vs {away}")
with st.container(horizontal=True, vertical_alignment="center"):
    st.badge(f"{date:%A %d %B %Y}", icon=":material/calendar_month:", color="gray")
    st.badge(f"Serie A {window.label}", icon=":material/sports_soccer:", color="blue")
    if mode == UPCOMING:
        st.badge("Future fixture, chosen by you", icon=":material/schedule:", color="green",
                 help="Any two clubs and any date left in the season can be chosen; this is not taken from the "
                      "official fixture list.")
    else:
        st.badge("Historical demonstration", icon=":material/history:", color="orange")
    if mode == UPCOMING and freshness.warnings():
        st.badge("Data may be out of date", icon=":material/warning:", color="red")

last_used = history["last_date"]
note = (f"The pre-match features use the {history['matches']:,} matches completed before this date, the latest on "
        f"{last_used:%d %b %Y}." if pd.notna(last_used) else "No earlier matches exist in the data.")
if mode == UPCOMING and today > latest_date:
    note += (f" Matches played after {latest_date:%d %b %Y} are not in the dataset, so recent results, injuries "
             "and suspensions are not reflected.")
st.caption(note)
if mode == UPCOMING:
    for message in freshness.warnings():
        st.warning(message, icon=":material/update:", title="Stale match data")

predictions_tab, goals_tab, research_tab = st.tabs(
    [":material/query_stats: Match predictions", ":material/sports_soccer: Goals & BTTS history",
     ":material/science: Research & methodology"], key="section")

# ================================================================ 1. match predictions

with predictions_tab:
    st.caption(f"**Model output.** Each probability comes from one of three saved machine-learning models, trained "
               f"on Serie A {training_label} and never retrained by the app. In the historical backtest the "
               "bookmaker's odds were more accurate than all three (see Research & methodology).")

    if mode == DEMO:
        actual = played_fixture(matches, home, away, date)
        if actual is not None:
            st.info(f"{home} {int(actual['home_goals'])}–{int(actual['away_goals'])} {away} "
                    f"({OUTCOME_NAMES[str(actual['result'])].lower()}). The prediction below uses only results from "
                    "before this day.", title="Actual result", icon=":material/scoreboard:")
            if training and window.season_start <= training[1]:
                st.warning(f"This match is part of the models' training data ({training_label}), so this is a "
                           "demonstration, not a genuine out-of-sample forecast.", icon=":material/warning:")
            else:
                st.caption(f"{window.label} is after the training seasons ({training_label}): this fixture was not "
                           "used to fit or tune the models, only to measure their accuracy afterwards.")
        else:
            st.caption("These teams did not meet on this date in the data: this is a hypothetical fixture.")

    # Per-model cards
    st.subheader("Predictions by model")
    cards = st.columns(len(models))
    for column, (label, probs) in zip(cards, probabilities.iterrows()):
        with column.container(border=True, height="stretch"):
            best = most_likely(probs)
            st.markdown(f"**{label}**")
            st.metric("Most likely outcome", OUTCOME_NAMES[best], delta=f"{probs[best]:.1%}", delta_color="off",
                      delta_arrow="off")
            table = outcome_table(probs)
            table["Fair odds"] = table["Fair odds"].where(table["Probability"] > 0)  # 1/0 shown as blank
            table["Probability"] = table["Probability"] * 100  # shown as a percentage with one decimal
            st.dataframe(
                table,
                hide_index=True,
                column_config={
                    "Outcome": st.column_config.TextColumn("Outcome"),
                    "Probability": st.column_config.NumberColumn("Probability", format="%.1f%%"),
                    "Fair odds": st.column_config.NumberColumn("Fair odds", format="%.2f",
                                                               help="1 / probability, with no bookmaker margin"),
                    "Most likely": None,  # shown by the metric above the table
                },
                alt=f"{label}: probability and fair odds for each outcome",
            )

    # Comparison charts, full width and stacked so that neither is squeezed
    st.subheader("Model comparison")
    show_average = st.toggle("Show the simple average of the three models", value=False,
                             help="An unweighted mean of the three models. It was not trained or validated "
                                  "as a separate model.")
    average = simple_average(probabilities)
    shown = probabilities.copy()
    if show_average:
        shown.loc[AVERAGE] = average
    model_order = list(shown.index)

    split = shown.rename(columns=OUTCOME_NAMES).rename_axis("Model").reset_index().melt(
        id_vars="Model", var_name="Outcome", value_name="Probability")
    split["order"] = split["Outcome"].map({name: i for i, name in enumerate(OUTCOME_ORDER)})
    split = split.sort_values(["Model", "order"])
    split["end"] = split.groupby("Model")["Probability"].cumsum()
    split["start"] = split["end"] - split["Probability"]
    split["middle"] = (split["start"] + split["end"]) / 2
    split["label"] = split["Probability"].map("{:.0%}".format)
    split_base = alt.Chart(split).encode(
        y=alt.Y("Model:N", sort=model_order, title=None, axis=alt.Axis(labelLimit=220, labelFontSize=12)),
        tooltip=["Model", "Outcome", alt.Tooltip("Probability:Q", format=".1%")],
    )
    split_chart = (
        split_base.mark_bar(height={"band": 0.8}, stroke=SURFACE, strokeWidth=2).encode(
            x=alt.X("start:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None, tickCount=5)),
            x2="end:Q",
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=OUTCOME_ORDER, range=list(OUTCOME_COLORS.values())),
                            legend=alt.Legend(orient="top", title=None, labelFontSize=12)))
        + split_base.transform_filter("datum.Probability >= 0.07").mark_text(fontSize=12, fontWeight="bold").encode(
            x="middle:Q", text="label:N",
            color=alt.condition(alt.datum.Outcome == OUTCOME_NAMES["D"], alt.value("#0b0b0b"), alt.value("#ffffff")))
    ).properties(height=64 * len(model_order) + 40, title="How each model splits the probability")
    st.altair_chart(split_chart, width="stretch",
                    alt=f"Stacked bars of home win, draw and away win probabilities for {home} vs {away}, one bar per "
                        "model")

    dots = shown.rename(columns=OUTCOME_NAMES).rename_axis("Model").reset_index().melt(
        id_vars="Model", var_name="Outcome", value_name="Probability")
    ranges = dots[dots["Model"] != AVERAGE].groupby("Outcome")["Probability"].agg(["min", "max"]).reset_index()
    low = max(0.0, dots["Probability"].min() - 0.05)
    high = min(1.0, dots["Probability"].max() + 0.05)
    x_scale = alt.Scale(domain=[low, high], zero=False, nice=False)
    y_outcome = alt.Y("Outcome:N", sort=OUTCOME_ORDER, title=None, axis=alt.Axis(labelFontSize=12))
    range_rules = alt.Chart(ranges).mark_rule(strokeWidth=10, color="#e1e0d9", strokeCap="round").encode(
        y=y_outcome, x=alt.X("min:Q", scale=x_scale), x2="max:Q",
        tooltip=["Outcome", alt.Tooltip("min:Q", format=".1%", title="Lowest"),
                 alt.Tooltip("max:Q", format=".1%", title="Highest")])
    model_dots = alt.Chart(dots).mark_point(filled=True, size=170, stroke=SURFACE, strokeWidth=1.5, opacity=1).encode(
        y=y_outcome,
        x=alt.X("Probability:Q", scale=x_scale,
                axis=alt.Axis(format="%", title="Predicted probability (axis zoomed to the models' range)")),
        color=alt.Color("Model:N", scale=alt.Scale(domain=model_order, range=[MODEL_COLORS[m] for m in model_order]),
                        legend=alt.Legend(orient="top", title=None, labelFontSize=12, labelLimit=260, symbolSize=140,
                                          columns=2)),
        shape=alt.Shape("Model:N", scale=alt.Scale(domain=model_order,
                                                    range=["circle", "square", "triangle-up", "diamond"][
                                                          :len(model_order)]), legend=None),
        tooltip=["Model", "Outcome", alt.Tooltip("Probability:Q", format=".1%")],
    )
    disagreement_chart = (range_rules + model_dots).properties(
        height=80 * len(OUTCOMES) + 90, title="Where the models disagree")
    st.altair_chart(disagreement_chart, width="stretch",
                    alt="Dot plot of each model's probability for home win, draw and away win; the grey band spans "
                        "the lowest to the highest model")

    spread = (probabilities[OUTCOMES].max() - probabilities[OUTCOMES].min()).rename(OUTCOME_NAMES)
    st.caption("Largest disagreement between the models: "
               + ", ".join(f"{name.lower()} {value * 100:.1f} points" for name, value in spread.items()) + ".")
    if show_average:
        st.caption(f"Simple average: home win {average['H']:.1%}, draw {average['D']:.1%}, away win "
                   f"{average['A']:.1%} (unvalidated; not a separately trained model).")

    with st.expander("Pre-match features for this fixture", icon=":material/table_view:"):
        used = set(next(iter(models.values())).feature_columns)
        rows = pd.DataFrame({
            "Feature": TEAM_FEATURES,
            home: [features[f"home_{name}"].iloc[0] for name in TEAM_FEATURES],
            away: [features[f"away_{name}"].iloc[0] for name in TEAM_FEATURES],
            "Used by the saved models": [all(f"{side}_{name}" in used for side in SIDES) for name in TEAM_FEATURES],
        })
        for team in (home, away):
            rows[team] = rows[team].map(lambda v: "missing" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float)
                                                                                else str(v)))
        st.dataframe(rows, hide_index=True, alt=f"The 25 pre-match features for {home} and {away}")
        st.caption("form = last 5 or 10 Serie A matches (across seasons, within a continuous Serie A spell); "
                   "venue_form = home matches only for the home team, away matches only for the away team; "
                   "season_ = this season so far; prev_season_ = last season; ppg = points per game, gf / ga / gd = "
                   "goals for / against / difference per game. 'missing' means not enough history yet.")

    with st.expander("About the models and their limitations", icon=":material/info:"):
        st.markdown(f"""
**Training period.** The saved models were trained on Serie A seasons {training_label}. Later completed
matches are used only to build the pre-match features (form, season-to-date record); the models were
not retrained on them. Latest completed match in the data: **{latest_date:%d %B %Y}**.

**Features.** 50 engineered pre-match features, 25 per team: recent form over the last 5 and 10 matches,
home or away form, this season's record so far, last season's record, experience in Serie A, and a
promoted-team flag. They are computed only from matches completed before the fixture date. The saved
models use the 22 features of the compact set (marked in the table above). Elo ratings are **not** used.

**Why the models differ.** Logistic Regression combines the features in a straight line; Random Forest
and XGBoost combine many decision trees that can capture non-linear effects. Each was tuned separately,
so they weigh the same information slightly differently. Historically none of them was clearly better
than the others.

**Known limitations.**
- Draws are hard to predict: the models rarely make a draw their most likely outcome.
- No team news: injuries, suspensions, transfers, line-ups and motivation are not in the data, which is
  one reason bookmaker odds were more accurate than these models in the historical backtest.
- Promoted clubs have little or no Serie A history, so their features are partly missing.
- Matches played after the latest data date are not included.
""")

# ================================================================ 2. goals & BTTS history
# Historical frequencies only (src/goal_stats.py). Nothing here feeds into, or comes from, the 1X2 models.

with goals_tab:
    st.header("Goals & BTTS Analysis")
    st.caption(f"How many goals each club's own Serie A matches produced before {date:%d %B %Y}. These are "
               "historical frequencies, not model predictions: they are not probabilities for this fixture and have "
               "not been validated as forecasts.")
    st.info("The goals models we tested (Poisson, direct and adaptive event models) have not met their readiness "
            "rules, so the app shows no Over/Under or BTTS forecast. Research & methodology explains why.",
            icon=":material/history_edu:", title="Descriptive history, not a forecast")

    WINDOW_KEYS = {w.label: w.key for w in HISTORY_WINDOWS.values()}
    window_label = st.segmented_control(
        "Historical window", list(WINDOW_KEYS), default="Last 10 matches", required=True, key="goal_window",
        help="Only matches completed before the match date are used. Home / away windows use the home club's home "
             "matches and the away club's away matches. Windows run across seasons within a club's current Serie A "
             "spell; promoted and returning clubs start with no history.")
    goals = fixture_goal_stats(load_goal_rows(), home, away, date, WINDOW_KEYS[window_label], window.season_start)
    samples = (goals.home, goals.away)
    TEAM_ORDER = [home, away]
    TEAM_COLORS = {home: HOME_COLOR, away: AWAY_COLOR}  # fixed per club, so a club without data never repaints the other
    with_data = [sample.team for sample in samples if sample.size]
    TEAM_SCALE = alt.Scale(domain=with_data, range=[TEAM_COLORS[team] for team in with_data])
    REST_COLOR = "#d9d8d3"  # the "No" / "Under" part of a split bar

    def sample_period(sample) -> str:
        seasons = " and ".join(season_name(s) for s in sample.seasons) if len(sample.seasons) <= 2 else \
            f"{season_name(sample.seasons[0])} to {season_name(sample.seasons[-1])}"
        first, last = sample.matches["date"].iloc[0], sample.matches["date"].iloc[-1]
        return f"{first:%d %b %Y} to {last:%d %b %Y} ({seasons})"

    st.subheader("At a glance")
    for column, sample in zip(st.columns(2), samples):
        with column.container(border=True, height="stretch"):
            st.markdown(f"**{sample.team}** · {sample.venue} team · {sample.label.lower()}")
            if sample.size == 0:
                st.metric(sample.label, sample.sample_text())
                st.warning(f"{sample.team}: {sample.empty_reason()}. Its statistics are shown as missing, not as 0%.",
                           icon=":material/block:")
            else:
                stats = sample.stats
                st.caption(f"Sample: {sample.sample_text()}")
                first, second, third = st.columns(3)
                first.metric("Goals per match", f"{stats['total_goals']:.2f}",
                             help="Both clubs' goals in this club's matches")
                second.metric("Over 2.5 goals", f"{rate(stats, line_key(2.5, 'over')):.0%}",
                              help=f"{stats[line_key(2.5, 'over')]} of {stats['matches']} matches")
                third.metric("Both teams scored", f"{rate(stats, 'btts_yes'):.0%}",
                             help=f"{stats['btts_yes']} of {stats['matches']} matches")
                if not sample.complete:
                    st.badge(f"Short sample: only {sample.size} of {sample.target} matches available",
                             icon=":material/warning:", color="orange")
                st.caption(f"Matches from {sample_period(sample)}.")
    smallest = min((sample.size for sample in samples if sample.size), default=0)
    if smallest:
        st.caption(f"Small samples swing a lot: with {smallest} {'match' if smallest == 1 else 'matches'}, one "
                   f"result moves a percentage by {100 / smallest:.0f} points.")

    if not any(sample.size for sample in samples):
        st.info("Neither club has matches in this window before this date, so there are no goal statistics to show.",
                icon=":material/info:")
    else:
        def split_rows(items):
            """Rows for a split bar: the share of matches with the event (team colour) and without it (grey)."""
            records = []
            for sample in samples:
                stats, n = sample.stats, sample.stats["matches"]
                if not n:
                    continue
                for category, key, yes_label, no_label in items:
                    yes = stats[key]
                    records.append({"Team": sample.team, "Category": category, "Share": yes / n,
                                    "Rest": (n - yes) / n, "Yes": f"{yes_label}: {yes} of {n} ({yes / n:.0%})",
                                    "No": f"{no_label}: {n - yes} of {n} ({(n - yes) / n:.0%})"})
            return pd.DataFrame(records)

        def split_chart(data, categories, title):
            base = alt.Chart(data).encode(y=alt.Y("Category:N", sort=categories, title=None,
                                                  axis=alt.Axis(labelFontSize=12, labelLimit=180)),
                                          yOffset=alt.YOffset("Team:N", sort=TEAM_ORDER),
                                          tooltip=["Team", alt.Tooltip("Yes:N", title="Yes / Over"),
                                                   alt.Tooltip("No:N", title="No / Under")])
            x_axis = alt.X("Share:Q", scale=alt.Scale(domain=[0, 1]),
                           axis=alt.Axis(format="%", title="Share of the club's matches", tickCount=5))
            yes = base.mark_bar(height={"band": 0.85}).encode(
                x=x_axis, color=alt.Color("Team:N", scale=TEAM_SCALE,
                                          legend=alt.Legend(orient="top", title=None, labelFontSize=12)))
            no = base.mark_bar(height={"band": 0.85}, color=REST_COLOR, stroke=SURFACE, strokeWidth=2,
                               cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(x="Share:Q", x2=alt.datum(1))
            yes_text = base.transform_filter("datum.Share >= 0.12").mark_text(
                align="left", baseline="middle", dx=5, fontSize=12, color="#ffffff", fontWeight="bold").encode(
                x=alt.datum(0), text=alt.Text("Share:Q", format=".0%"))
            no_text = base.transform_filter("datum.Rest >= 0.12").mark_text(
                align="right", baseline="middle", dx=-5, fontSize=12, color=INK_2).encode(
                x=alt.datum(1), text=alt.Text("Rest:Q", format=".0%"))
            return (no + yes + yes_text + no_text).properties(height=len(categories) * 76 + 90, title=title)

        GOAL_AXIS = {"goals_for": "Scored", "goals_against": "Conceded", "total_goals": "Total (both clubs)"}
        averages = pd.DataFrame([
            {"Team": sample.team, "Statistic": GOAL_AXIS[key], "Value": sample.stats[key],
             "Sample": sample.sample_text()}
            for sample in samples if sample.size for key in AVERAGES
        ])
        top = max(averages["Value"].max(), 0.5) * 1.18
        goal_base = alt.Chart(averages).encode(
            y=alt.Y("Statistic:N", sort=list(GOAL_AXIS.values()), title=None,
                    axis=alt.Axis(labelFontSize=12, labelLimit=180)),
            yOffset=alt.YOffset("Team:N", sort=TEAM_ORDER),
            x=alt.X("Value:Q", scale=alt.Scale(domain=[0, top]), axis=alt.Axis(title="Goals per match")),
            tooltip=["Team", "Statistic", alt.Tooltip("Value:Q", format=".2f", title="Goals per match"), "Sample"],
        )
        goals_chart = (goal_base.mark_bar(cornerRadiusEnd=4, height={"band": 0.85}).encode(
            color=alt.Color("Team:N", scale=TEAM_SCALE, legend=alt.Legend(orient="top", title=None, labelFontSize=12)))
            + goal_base.mark_text(align="left", baseline="middle", dx=4, fontSize=12, color=INK_2).encode(
                text=alt.Text("Value:Q", format=".2f"))
        ).properties(height=len(AVERAGES) * 76 + 90, title="Goals scored and conceded per match")

        scoring_items = [("Both teams scored", "btts_yes", "BTTS Yes", "BTTS No"),
                         ("Club scored", "scored", "Scored", "Failed to score"),
                         ("Clean sheet", "clean_sheet", "Clean sheet", "Conceded")]
        line_items = [(f"{line:g} goals", line_key(line, "over"), f"Over {line:g}", f"Under {line:g}")
                      for line in GOAL_LINES]
        scoring = split_rows(scoring_items)

        st.subheader("Goals, scoring and goal lines")
        left, right = st.columns(2, gap="large")
        with left:
            st.altair_chart(goals_chart, width="stretch",
                            alt=f"Bar chart of goals scored, conceded and in total per match for {home} and {away}")
        with right:
            st.altair_chart(split_chart(scoring, [item[0] for item in scoring_items],
                                        "BTTS, scoring and clean sheets"), width="stretch",
                            alt=f"Split bars of the share of BTTS, scoring and clean-sheet matches for {home} and "
                                f"{away}")

        left, right = st.columns(2, gap="large")
        with left:
            st.altair_chart(split_chart(split_rows(line_items), [item[0] for item in line_items],
                                        "Over (colour) / Under (grey) each goal line"),
                            width="stretch", alt=f"Split bars of the share of matches over and under 0.5, 1.5, 2.5 "
                                                 f"and 3.5 total goals for {home} and {away}")
        with right:
            def cell(sample, key):
                stats = sample.stats
                if not stats["matches"]:
                    return "n/a"
                if key in AVERAGES:
                    return f"{stats[key]:.2f}"
                return f"{stats[key]} of {stats['matches']} ({rate(stats, key):.0%})"

            table = pd.DataFrame({
                "Statistic": list(AVERAGES.values()) + list(EVENTS.values()),
                f"{home} (home)": [cell(goals.home, k) for k in list(AVERAGES) + list(EVENTS)],
                f"{away} (away)": [cell(goals.away, k) for k in list(AVERAGES) + list(EVENTS)],
            })
            st.dataframe(table, hide_index=True, height="stretch",
                         alt=f"Goal statistics table for {home} and {away}")

        st.caption("Over / Under count the goals of both clubs in each of the club's matches; BTTS Yes means both "
                   "clubs scored at least once. Each percentage is a count divided by that club's number of matches. "
                   "'n/a' means no matches in the window, not 0%.")

        with st.expander("Matches in each sample", icon=":material/list:"):
            for column, sample in zip(st.columns(2), samples):
                with column:
                    st.markdown(f"**{sample.team}**: {sample.label.lower()} ({sample.sample_text()})")
                    if sample.size == 0:
                        st.caption("No matches.")
                        continue
                    listed = sample.matches.iloc[::-1]
                    st.dataframe(pd.DataFrame({
                        "Date": listed["date"].dt.strftime("%d %b %Y"),
                        "Venue": listed["venue"].str.capitalize(),
                        "Opponent": listed["opponent"],
                        "Score": listed["goals_for"].astype(str) + "–" + listed["goals_against"].astype(str),
                        "Total goals": listed["total_goals"],
                        "BTTS": listed["btts"].map({True: "Yes", False: "No"}),
                    }), hide_index=True, alt=f"{sample.team}'s matches in this window, newest first")
            st.caption("Score is shown from the club's point of view (its goals first).")

# ================================================================ 3. research & methodology
# Summaries of finished experiments, read from the saved result files. Nothing is recomputed or retrained.

with research_tab:
    st.header("Research & methodology")
    st.caption("What the historical experiments found, and why the app shows what it shows. Numbers are read from "
               "the saved result files in outputs/; the app never reruns an experiment or retrains a model.")
    root = str(support.RESEARCH_ROOT)

    def research(name: str):
        """A summary, or None after showing why it is unavailable."""
        try:
            return load_research(name, root)
        except support.ResearchFileMissing as missing:
            st.info(f"{missing.what} results are not available in this copy of the project (`{missing.path.name}` "
                    f"is missing from `{missing.path.parent.name}/`). They are git-ignored and created by "
                    f"`{missing.command}`; see the README. The rest of the app is unaffected.",
                    icon=":material/folder_off:", title="Saved results not found")
            return None

    st.subheader("Key findings")
    st.markdown("""
- **The three 1X2 models perform about the same** in the 2018-19 to 2024-25 walk-forward backtest.
- **Bookmaker odds beat all three models** over the same fixtures, probably partly because they reflect
  team news that the models do not have.
- **Elo ratings gave small improvements** but did not show robust extra predictive skill, so they are not used.
- **Several goals models estimate expected goals better than the league average**, but none of their Over/Under
  or BTTS probabilities met the pre-specified readiness rules.
- **Adaptive scoring rates are promising** (the best goals-market scores so far) but are not validated enough
  to be deployed.
""")

    with st.container(border=True):
        st.markdown("**How the models were tested: walk-forward backtesting**")
        st.markdown("""
For each season from 2018-19 to 2024-25, every model was tuned and trained only on earlier seasons and then
scored on that season's matches, as if predicting it in real time. **Log loss** measures how much probability
a forecast gave to what actually happened (lower is better; a 1X2 coin toss scores about 1.10). Decision rules
were written down before each experiment was run, and each was judged on all seven seasons.
""")

    st.subheader("1X2 models versus the bookmaker")
    outcome = research("outcome")
    if outcome is not None:
        view = outcome.rename(columns={"matches": "Matches", "log_loss": "Log loss", "brier": "Brier score",
                                       "accuracy": "Accuracy", "seasons_best": "Seasons with the best log loss"})
        view = view.drop(columns="seasons").rename_axis("Forecast").reset_index()
        view["Accuracy"] = view["Accuracy"] * 100
        st.dataframe(view, hide_index=True, column_config={
            "Log loss": st.column_config.NumberColumn(format="%.4f", help="Lower is better"),
            "Brier score": st.column_config.NumberColumn(format="%.4f", help="Lower is better"),
            "Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
        }, alt="Pooled walk-forward scores of the three models, the bookmaker and a baseline")
        models_only = outcome.loc[[m for m in probabilities.index if m in outcome.index], "log_loss"]
        bookmaker = [name for name in outcome.index if name.startswith("Bookmaker")]
        if len(models_only) and bookmaker:
            bookie = outcome.loc[bookmaker[0]]
            st.caption(f"{int(outcome['matches'].iloc[0]):,} fixtures with odds, {int(bookie['seasons'])} seasons. "
                       f"The three models are within {models_only.max() - models_only.min():.4f} log loss of each "
                       f"other; the bookmaker is {models_only.min() - bookie['log_loss']:.4f} better than the best "
                       f"model and had the lowest log loss in {int(bookie['seasons_best'])} of "
                       f"{int(bookie['seasons'])} seasons. 'Baseline' always predicts the historical outcome shares.")

    st.subheader("Elo ratings")
    elo = research("elo")
    if elo is not None:
        view = elo.rename_axis("Model").reset_index().rename(columns={
            "log_loss": "Log loss", "log_loss_with_elo": "With Elo", "difference": "Change",
            "seasons_improved": "Seasons improved", "seasons": "Seasons"})
        st.dataframe(view, hide_index=True, column_config={
            "Log loss": st.column_config.NumberColumn(format="%.4f"),
            "With Elo": st.column_config.NumberColumn(format="%.4f"),
            "Change": st.column_config.NumberColumn(format="%+.4f", help="Negative = Elo helped"),
        }, alt="Pooled log loss of each model with and without Elo features")
        st.caption("Adding leakage-safe Elo ratings lowered every model's log loss slightly, but in the "
                   "pre-specified test no improvement was distinguishable from chance after correcting for multiple "
                   "comparisons, so Elo is **not** a production feature.")

    st.subheader("Goals models")
    expected = research("expected_goals")
    if expected is not None:
        view = expected.rename_axis("Method").reset_index().rename(columns={
            "matches": "Matches", "deviance": "Poisson deviance", "mae_home": "Error, home goals",
            "mae_away": "Error, away goals"})
        st.dataframe(view, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.4f", help="Lower is better")
            for c in ("Poisson deviance", "Error, home goals", "Error, away goals")},
            alt="Expected-goals accuracy of the Poisson methods")
        st.caption("Expected goals per side: the Poisson regression and the team attack/defence model are clearly "
                   "better than the league average. Errors are mean absolute errors in goals.")

    events = research("events")
    if events is not None:
        event_table, source = events
        view = event_table.rename_axis("Method").reset_index()
        st.dataframe(view, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.4f", help="Lower is better")
            for c in view.columns if c.endswith("log loss")},
            alt="Over 2.5 and BTTS log loss of every goals-market method")
        st.caption("Over 2.5 and BTTS probabilities, same fixtures, 2018-19 to 2024-25. 'Event frequency' is the "
                   "static historical share every method must beat. Differences are a few thousandths of a "
                   "log-loss unit, and none of the methods met its readiness rule, so the app shows no goals "
                   "forecast." + (" The adaptive methods are missing because their results were not found."
                                  if source != "scoring_drift" else ""))
        if source == "scoring_drift":
            st.caption("**Adaptive scoring rates** follow the last 200 league matches. They have the lowest pooled "
                       "log loss for both events and a clear gain over the static frequency for Over 2.5, but the "
                       "Over 2.5 forecasts were too timid (calibration) and BTTS gains were not consistent. These "
                       "seasons also shaped the idea, so the result needs confirmation on seasons not yet used.")

    st.subheader("Decisions")
    st.dataframe(support.DECISIONS, hide_index=True, alt="Research questions, evidence, outcomes and what the app uses",
                 column_config={"Outcome": st.column_config.TextColumn(width="large")})

    st.subheader("Saved models on recent seasons")
    holdout = research("holdout")
    if holdout is not None:
        view = holdout.rename(columns={"split": "Season", "model": "Model", "matches": "Matches",
                                       "log_loss": "Log loss", "brier": "Brier score", "accuracy": "Accuracy"})
        view["Season"] = view["Season"].map({"validation": "2025-26 (validation)",
                                             "test": "2026-27 (holdout, so far)"}).fillna(view["Season"])
        view["Accuracy"] = view["Accuracy"] * 100
        st.dataframe(view, hide_index=True, column_config={
            "Log loss": st.column_config.NumberColumn(format="%.4f"),
            "Brier score": st.column_config.NumberColumn(format="%.4f"),
            "Accuracy": st.column_config.NumberColumn(format="%.1f%%")},
            alt="Scores of the saved models on the validation and holdout seasons")
        test_matches = int(holdout.loc[holdout["split"] == "test", "matches"].max()) if (holdout["split"] == "test").any()             else 0
        st.caption("The 2025-26 scores come from candidate models trained up to 2024-25; that season was used to "
                   f"choose the settings, after which the final models were refitted on {training_label}. The 2026-27 "
                   f"rows are a genuine holdout, but only {test_matches} matches, too few for firm conclusions.")

    st.subheader("Limitations")
    st.markdown(f"""
- **Historical demos are not forecasts.** A historical fixture is rebuilt from the results before its date, but
  if its season was used for training ({training_label}) or for choosing settings, the models have effectively
  seen it. Only fixtures after the latest training and validation data are genuine out-of-sample predictions.
- **No team news.** Injuries, suspensions, line-ups, transfers and motivation are not in the data.
- **Data freshness.** Predictions use only matches up to {latest_date:%d %B %Y}; anything played later is missing.
- **No fixture list.** The app does not know the official schedule: you choose the clubs and the date.
- **Sample size.** Seven backtest seasons cannot resolve differences of a few thousandths of a log-loss unit.
- **Not betting advice.** Fair odds are 1 / probability without a bookmaker margin; no betting strategy was tested.
""")

st.info("These are statistical estimates from historical data, not guarantees and not betting recommendations. "
        "Fair odds are simply 1 / probability, without any bookmaker margin.",
        icon=":material/gavel:", title="Disclaimer")

"""Serie A Match Predictor: an interactive Streamlit app.

Run from the project folder (after generating the data and models, see the README)::

    streamlit run app.py

The app never retrains anything. It reads the processed match data and the
models saved by ``python -m src.train``, builds the 50 pre-match features of
the chosen fixture from matches completed strictly before its date
(``src/predict.py``), and shows each model's probabilities.
"""

import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluate import OUTCOME_NAMES, OUTCOMES
from src.feature_engineering import SIDES, TEAM_FEATURES
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
MODEL_COLORS = {"Logistic Regression": "#2a78d6", "Random Forest": "#eb6834", "XGBoost": "#1baf7a", AVERAGE: "#a3a29b"}
OUTCOME_ORDER = [OUTCOME_NAMES[o] for o in OUTCOMES]


# ---------------------------------------------------------------- cached loading


@st.cache_data(show_spinner="Loading match data...")
def load_data() -> pd.DataFrame:
    return load_matches()


@st.cache_resource(show_spinner="Loading saved models...")
def load_models():
    return load_saved_models()


@st.cache_data(max_entries=256, show_spinner=False)
def cached_fixture_features(home: str, away: str, date: dt.date) -> pd.DataFrame:
    return fixture_features(load_data(), home, away, date)


def next_saturday(start: dt.date) -> dt.date:
    return start + dt.timedelta(days=(5 - start.weekday()) % 7)


def keep_valid(key: str, options, fallback):
    """Reset a widget's stored value when it is no longer one of its options."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = fallback


# ---------------------------------------------------------------- data and models

st.title("Serie A Match Predictor")
st.caption("Home win, draw and away win probabilities from three machine-learning models trained on Serie A "
           "results, with fair odds and a side-by-side model comparison.")

try:
    matches = load_data()
except FileNotFoundError as error:
    st.error(f"{error}\n\nRun `python -m src.data_loader` and `python -m src.preprocessing` first (see the README).",
             title="Match data not found")
    st.stop()

try:
    models = load_models()
except ModelFilesMissing as error:
    st.error(str(error), title="Saved models not found")
    st.stop()

windows = season_windows(matches)
latest_season = windows[max(windows)]
latest_date = latest_completed_date(matches).date()
training = next(iter(models.values())).training_seasons
training_label = f"{season_name(training[0])} to {season_name(training[1])}" if training else "unknown"

# ---------------------------------------------------------------- controls

with st.sidebar:
    st.header("Fixture", divider="gray")
    mode = st.segmented_control("Mode", [UPCOMING, DEMO], default=UPCOMING, required=True, key="mode",
                                help="Historical demo rebuilds any past fixture using only the results "
                                     "known before that day.")

    if mode == UPCOMING:
        window = latest_season
        min_date = latest_date + dt.timedelta(days=1)
        max_date = window.last_date.date()
        default_date = min(next_saturday(max(dt.date.today(), min_date)), max_date)
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
    st.metric("Latest completed match in the data", f"{latest_date:%d %b %Y}")
    st.caption(f"{len(matches):,} completed Serie A matches from {season_name(min(windows))} to "
               f"{latest_season.label}. Clubs listed for {window.label} come from that season's matches.")
    if mode == UPCOMING:
        st.caption(f"Upcoming fixtures can be chosen from {min_date:%d %b %Y} to the end of {window.label} "
                   f"({max_date:%d %b %Y}).")

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
        st.badge("Upcoming fixture", icon=":material/schedule:", color="green")
    else:
        st.badge("Historical demonstration", icon=":material/history:", color="orange")

last_used = history["last_date"]
note = (f"The pre-match features use the {history['matches']:,} matches completed before this date, the latest on "
        f"{last_used:%d %b %Y}." if pd.notna(last_used) else "No earlier matches exist in the data.")
if mode == UPCOMING and dt.date.today() > latest_date:
    note += (f" Matches played after {latest_date:%d %b %Y} are not in the dataset, so recent results, injuries "
             "and suspensions are not reflected.")
st.caption(note)

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
                "Probability": st.column_config.ProgressColumn("Probability", format="%.1f%%", min_value=0,
                                                               max_value=100),
                "Fair odds": st.column_config.NumberColumn("Fair odds", format="%.2f",
                                                           help="1 / probability, with no bookmaker margin"),
                "Most likely": st.column_config.CheckboxColumn("Most likely"),
            },
            alt=f"{label}: probability and fair odds for each outcome",
        )

# Comparison charts
st.subheader("Model comparison")
show_average = st.toggle("Show the simple average of the three models", value=False,
                         help="An unweighted mean of the three models. It was not trained or validated "
                              "as a separate model.")
long = probabilities.rename(columns=OUTCOME_NAMES).rename_axis("Model").reset_index().melt(
    id_vars="Model", var_name="Outcome", value_name="Probability")
average = simple_average(probabilities)
if show_average:
    long = pd.concat([long, pd.DataFrame({"Model": AVERAGE, "Outcome": OUTCOME_ORDER,
                                          "Probability": [average[o] for o in OUTCOMES]})], ignore_index=True)
model_order = list(probabilities.index) + ([AVERAGE] if show_average else [])
color = alt.Color("Model:N", scale=alt.Scale(domain=model_order, range=[MODEL_COLORS[m] for m in model_order]),
                  legend=alt.Legend(orient="bottom", title=None))

bars = alt.Chart(long).encode(
    y=alt.Y("Outcome:N", sort=OUTCOME_ORDER, title=None),
    yOffset=alt.YOffset("Model:N", sort=model_order),
    x=alt.X("Probability:Q", axis=alt.Axis(format="%", title="Predicted probability"), scale=alt.Scale(domain=[0, 1])),
    color=color,
    tooltip=["Model", "Outcome", alt.Tooltip("Probability:Q", format=".1%")],
)
probability_chart = (bars.mark_bar(cornerRadiusEnd=4, height={"band": 0.9})
                     + bars.mark_text(align="left", dx=4, fontSize=11).encode(
                         text=alt.Text("Probability:Q", format=".1%"), color=alt.value("#52514e"))
                     ).properties(height=80 * len(OUTCOMES) * (len(model_order) / 3) + 40,
                                  title="Probability of each outcome")

diffs = (probabilities[OUTCOMES] - average).rename(columns=OUTCOME_NAMES).rename_axis("Model").reset_index().melt(
    id_vars="Model", var_name="Outcome", value_name="Difference")
diff_order = list(probabilities.index)
diff_bars = alt.Chart(diffs).encode(
    y=alt.Y("Outcome:N", sort=OUTCOME_ORDER, title=None),
    yOffset=alt.YOffset("Model:N", sort=diff_order),
    x=alt.X("Difference:Q", axis=alt.Axis(format="+.0%", title="Difference from the three-model average")),
    color=alt.Color("Model:N", scale=alt.Scale(domain=diff_order, range=[MODEL_COLORS[m] for m in diff_order]),
                    legend=alt.Legend(orient="bottom", title=None)),
    tooltip=["Model", "Outcome", alt.Tooltip("Difference:Q", format="+.1%")],
)
difference_chart = (alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#52514e").encode(x="x:Q")
                    + diff_bars.mark_bar(cornerRadius=3, height={"band": 0.9})
                    ).properties(height=80 * len(OUTCOMES) + 40,
                                 title="Where the models disagree (percentage points)")

left, right = st.columns(2)
with left:
    st.altair_chart(probability_chart, width="stretch",
                    alt=f"Grouped bar chart of home win, draw and away win probabilities for {home} vs {away} "
                        "by model")
with right:
    st.altair_chart(difference_chart, width="stretch",
                    alt="Bar chart of each model's probability minus the three-model average, by outcome")

spread = (probabilities[OUTCOMES].max() - probabilities[OUTCOMES].min()).rename(OUTCOME_NAMES)
st.caption("Largest disagreement between the models: "
           + ", ".join(f"{name.lower()} {value * 100:.1f} points" for name, value in spread.items()) + ".")
if show_average:
    st.caption(f"Simple average: home win {average['H']:.1%}, draw {average['D']:.1%}, away win {average['A']:.1%} "
               "(unvalidated; not a separately trained model).")

# ---------------------------------------------------------------- details

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
**Training period.** The saved models were trained on Serie A seasons {training_label}. Completed 2026-27
matches are used only to build the pre-match features (form, season-to-date record); the models were
not retrained on them. Latest completed match in the data: **{latest_date:%d %B %Y}**.

**Features.** 50 engineered pre-match features, 25 per team: recent form over the last 5 and 10 matches,
home or away form, this season's record so far, last season's record, experience in Serie A, and a
promoted-team flag. They are computed only from matches completed before the fixture date. The saved
models use the 22 features of the compact set (marked in the table above).

**Why the models differ.** Logistic Regression combines the features in a straight line; Random Forest
and XGBoost combine many decision trees that can capture non-linear effects. Each was tuned separately,
so they weigh the same information slightly differently. Historically none of them was clearly better
than the others (see notebooks 04, 06 and 07).

**Known limitations.**
- Draws are hard to predict: the models rarely make a draw their most likely outcome.
- No team news: injuries, suspensions, transfers, line-ups and motivation are not in the data, which is
  one reason bookmaker odds were more accurate than these models in the historical backtest (notebook 06).
- Promoted clubs have little or no Serie A history, so their features are partly missing.
- Matches played after the latest data date are not included.
""")

st.info("These are statistical estimates from historical data, not guarantees and not betting recommendations. "
        "Fair odds are simply 1 / probability, without any bookmaker margin.",
        icon=":material/gavel:", title="Disclaimer")

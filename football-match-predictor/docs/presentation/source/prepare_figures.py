"""Presentation-ready figures and chart data for docs/presentation/serie_a_match_predictor.pptx.

Run from the project folder, after the data, models and experiments exist:

    python docs/presentation/source/prepare_figures.py

Everything is read from files that already exist: data/processed/ and the
saved results in outputs/. No model is trained and no experiment is rerun.
The Elo intervals are recomputed from the saved Elo predictions with the same
fixed-seed paired bootstrap as notebook 08, and checked against the published
values. The original figures in outputs/figures/ are never touched.

Writes:
    docs/presentation/figures/*.png         redrawn charts in the deck's colours
    docs/presentation/source/chart_data.json  values for the native PowerPoint charts
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PREDICTIONS_DIR
from src.elo_experiment import comparison_table, load_elo_experiment
from src.feature_engineering import load_features
from src.goals_backtest import MARKET, load_goals_backtest, market_predictions, pooled_metrics as goals_pooled
from src.preprocessing import load_matches
import src.app_support as support

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
DATA_FILE = HERE / "chart_data.json"

# The deck's palette (same as the app): outcomes and models
HOME, DRAW, AWAY = "#2A78D6", "#9AA1AB", "#EB6834"
MODEL_COLORS = {"Logistic Regression": "#1BAF7A", "Random Forest": "#4A3AA7", "XGBoost": "#C44E9B"}
NAVY, INK, MUTED, GRID = "#0B1F3A", "#1B2433", "#5B6577", "#E3E6EB"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 15, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 220,
})


def season_label(start: int) -> str:
    return f"{start}-{str(start + 1)[-2:]}"


def completed_seasons(matches: pd.DataFrame) -> pd.DataFrame:
    """Seasons with all their matches played (the in-progress season is excluded, as in notebook 01)."""
    counts = matches.groupby("season_start").size()
    latest = matches["season_start"].max()
    return matches[matches["season_start"] != latest] if counts[latest] < counts.drop(latest).min() else matches


# ---------------------------------------------------------------- native chart data


def outcome_shares(matches: pd.DataFrame) -> dict:
    shares = matches["result"].value_counts(normalize=True)
    return {"home": shares["H"], "draw": shares["D"], "away": shares["A"], "matches": int(len(matches)),
            "goals_per_match": float((matches["home_goals"] + matches["away_goals"]).mean())}


def shares_by_season(matches: pd.DataFrame) -> dict:
    done = completed_seasons(matches)
    table = pd.crosstab(done["season_start"], done["result"], normalize="index")
    return {"seasons": [season_label(s) for s in table.index],
            "home": table["H"].round(4).tolist(), "draw": table["D"].round(4).tolist(),
            "away": table["A"].round(4).tolist()}


def form_vs_result(features: pd.DataFrame) -> dict:
    """Notebook 02, figure 10: result shares by 10-match form difference (bins with at least 100 matches)."""
    diff = features["home_form10_ppg"] - features["away_form10_ppg"]
    binned = pd.cut(diff, np.arange(-2.0, 2.01, 0.25))
    table = pd.crosstab(binned, features["result"], normalize="index")[["H", "D", "A"]]
    table["matches"] = binned.value_counts()
    table = table[table["matches"] >= 100]
    return {"centres": [f"{interval.mid:+.2f}" for interval in table.index],
            "home": table["H"].round(4).tolist(), "draw": table["D"].round(4).tolist(),
            "away": table["A"].round(4).tolist()}


def goals_by_season(matches: pd.DataFrame) -> dict:
    done = completed_seasons(matches)
    table = done.groupby("season_start")[["home_goals", "away_goals"]].mean()
    return {"seasons": [season_label(s) for s in table.index],
            "home": table["home_goals"].round(3).tolist(), "away": table["away_goals"].round(3).tolist(),
            "total": (table["home_goals"] + table["away_goals"]).round(3).tolist()}


def backtest_summary() -> dict:
    table = support.outcome_model_summary()
    baseline = table.loc["Baseline", "log_loss"]
    rows = {}
    for name in table.index:
        rows[name] = {"log_loss": float(table.loc[name, "log_loss"]), "accuracy": float(table.loc[name, "accuracy"]),
                      "improvement": float(baseline - table.loc[name, "log_loss"]),
                      "seasons_best": int(table.loc[name, "seasons_best"])}
    predictions = pd.read_csv(OUTPUTS_DIR / "backtest" / "predictions.csv")
    base = predictions[predictions["forecast"] == "Baseline"].groupby("season")[["H", "D", "A"]].first()
    return {"forecasts": rows, "matches": int(table["matches"].iloc[0]),
            "baseline_probabilities": {season_label(int(s)): base.loc[s].round(3).to_dict() for s in base.index}}


def over_2_5_vs_coin_flip() -> dict:
    """Pooled Over 2.5 log loss of the commit-15 methods and the market, and their gain over a 50/50 forecast."""
    goals = load_goals_backtest()
    pooled = goals_pooled(goals.predictions)["log_loss_over_2_5"]  # every method on all 2,660 fixtures
    odds = pd.read_csv(OUTPUTS_DIR / "goals" / "market_odds.csv", parse_dates=["date"]).astype(
        {"date": "datetime64[ns]", "home_team": "string", "away_team": "string"})
    market = market_predictions(goals.predictions, odds)
    market = market[market["method"] == MARKET]  # the fixtures with market odds
    coin = float(np.log(2))
    methods = {name: float(pooled[name]) for name in pooled.index}
    methods["Market average"] = float(market["log_loss_over_2_5"].mean())
    return {"coin_flip": coin, "log_loss": methods, "gain_over_coin_flip": {k: coin - v for k, v in methods.items()},
            "market_matches": int(len(market)),
            "matches": int(goals.predictions.drop_duplicates(["date", "home_team", "away_team"]).shape[0])}


def expected_goals() -> dict:
    table = support.expected_goals_summary()
    league, model = table.loc["League average", "deviance"], table.loc["Poisson regression", "deviance"]
    return {"league_average": float(league), "poisson_regression": float(model),
            "improvement_pct": float((league - model) / league * 100)}


def adaptive_results() -> dict:
    table, source = support.event_summary()
    if source != "scoring_drift":
        raise RuntimeError("the adaptive scoring-rate results are missing (python -m src.scoring_drift_backtest)")
    return {name: {"over_2_5": float(table.loc[name, "Over 2.5 log loss"]), "btts": float(table.loc[name, "BTTS log loss"])}
            for name in table.index}


def model_agreement() -> tuple[pd.DataFrame, dict]:
    probs = {}
    for name, file in (("Logistic Regression", "logistic_regression"), ("Random Forest", "random_forest"),
                       ("XGBoost", "xgboost")):
        probs[name] = pd.read_csv(PREDICTIONS_DIR / f"{file}_validation_2025-26.csv")
    home = pd.DataFrame({name: frame["prob_home_win"] for name, frame in probs.items()})
    picks = pd.DataFrame({name: frame["predicted"] for name, frame in probs.items()})
    agree = float((picks.nunique(axis=1) == 1).mean())
    stats = {"corr_rf_lr": float(home["Logistic Regression"].corr(home["Random Forest"])),
             "corr_xgb_lr": float(home["Logistic Regression"].corr(home["XGBoost"])),
             "same_pick_all_three": agree, "matches": int(len(home))}
    return home, stats


# ---------------------------------------------------------------- redrawn figures


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def agreement_figure(home: pd.DataFrame, stats: dict) -> None:
    """Figure 21, one panel: XGBoost vs Logistic Regression home-win probabilities, 2025-26."""
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 0.9], [0, 0.9], color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.scatter(home["Logistic Regression"], home["XGBoost"], s=34, color=MODEL_COLORS["XGBoost"], alpha=0.75,
               edgecolors="white", linewidths=0.6, zorder=2)
    ax.text(0.04, 0.95, f"correlation {stats['corr_xgb_lr']:.2f}", transform=ax.transAxes, fontsize=17,
            color=INK, va="top", fontweight="bold")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    ax.set_aspect("equal")
    ticks = np.arange(0, 0.91, 0.2)
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_xlabel("Logistic Regression: home-win probability", fontsize=15)
    ax.set_ylabel("XGBoost: home-win probability", fontsize=15)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.03, f"each dot = one 2025-26 match ({stats['matches']})", transform=ax.transAxes, ha="right",
            fontsize=12, color=MUTED)
    save(fig, "05_model_agreement.png")


def elo_figure() -> dict:
    """Figure 43, log-loss panel: Elo minus control, 95% and Bonferroni-adjusted intervals."""
    predictions = load_elo_experiment().predictions
    table = comparison_table(predictions, metric="log_loss", n_resamples=10_000)
    models = list(MODEL_COLORS)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(models))[::-1]
    ax.axvline(0, color=INK, linewidth=1.3, zorder=1)
    for yi, model in zip(y, models):
        r = table.loc[model]
        color = MODEL_COLORS[model]
        ax.hlines(yi, r["bootstrap_adj_ci_low"], r["bootstrap_adj_ci_high"], color=color, linewidth=2, alpha=0.7,
                  zorder=2)
        ax.hlines(yi, r["bootstrap_ci_low"], r["bootstrap_ci_high"], color=color, linewidth=9, alpha=0.45, zorder=2)
        ax.scatter(r["difference"], yi, s=120, color=color, edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(f"{r['difference']:+.4f}".replace("-", "−"), (r["difference"], yi), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=14, color=INK)
    ax.set_yticks(y, models, fontsize=15, color=INK)
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.set_xlabel("Change in pooled log loss with Elo  (left of 0 = Elo better)", fontsize=14)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("thick bar: 95% interval      thin line: interval corrected for testing 3 models", loc="left",
                 fontsize=12.5, color=MUTED, pad=12)
    save(fig, "08_elo_intervals.png")
    return {model: {k: float(table.loc[model, k]) for k in ("difference", "bootstrap_ci_low", "bootstrap_ci_high",
                                                            "bootstrap_adj_ci_low", "bootstrap_adj_ci_high")}
            for model in models}


def main() -> None:
    matches = load_matches()
    features = load_features()
    home, agreement = model_agreement()
    agreement_figure(home, agreement)
    data = {
        "outcome_shares": outcome_shares(matches),
        "shares_by_season": shares_by_season(matches),
        "form_vs_result": form_vs_result(features),
        "goals_by_season": goals_by_season(matches),
        "backtest": backtest_summary(),
        "model_agreement": agreement,
        "elo": elo_figure(),
        "over_2_5": over_2_5_vs_coin_flip(),
        "expected_goals": expected_goals(),
        "adaptive": adaptive_results(),
        "latest_match": str(matches["date"].max().date()),
    }
    DATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {DATA_FILE} and the figures in {FIGURES}")


if __name__ == "__main__":
    main()

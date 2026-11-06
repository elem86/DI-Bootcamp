# Presentation: Serie A Match Predictor

**[serie_a_match_predictor.pptx](serie_a_match_predictor.pptx)**: 12 slides, 16:9, speaker notes on every slide, about 13 minutes. It follows the [presentation outline](../presentation_outline.md) and the [project story](../project_story.md).

## Slides and sources

| # | Slide | Visual | Source |
|---|---|---|---|
| 1 | Can 30 years of results predict a football match? | App screenshot | `figures/01_app_predictions_hero.png` |
| 2 | Home wins 45% of the time: the bar to beat | Native bar chart | Outcome shares of all 11,826 matches (`data/processed/matches.csv`) |
| 3 | Home advantage has been shrinking for 30 years | Native line chart | Outcome shares per completed season (as in figure 04) |
| 4 | No peeking: every feature comes from the past | Editable timeline and native line chart | Result shares by 10-match form difference (as in figure 10) |
| 5 | Three algorithms, almost the same answer | Redrawn scatter | `figures/05_model_agreement.png` (figure 21, one panel) |
| 6 | Testing like a forecaster, one season at a time | Editable walk-forward diagram | The evaluation design in [methodology.md](../methodology.md#walk-forward-backtesting) |
| 7 | The bookmaker still beats the models | Native bar chart | Pooled log loss from `outputs/backtest/` (as in figure 33) |
| 8 | Two good ideas that didn't make the cut | Redrawn interval plot | `figures/08_elo_intervals.png` (figure 43, log-loss panel) |
| 9 | We can estimate how many goals, not whether | Native bar chart | Pooled Over 2.5 log loss from `outputs/goals/` (as in figure 51) |
| 10 | Scoring drifts, and following it almost works | Native line chart | Home and away goals per season (as in figure 64) |
| 11 | An app that is honest about what it knows | Three app screenshots | `figures/11_app_*_tab.png` |
| 12 | What worked, what didn't, what we learned | Text | [project_story.md](../project_story.md#12-conclusions) |

## Presentation-ready figures

`figures/` holds the edited visuals; the originals in `outputs/figures/` and `docs/screenshots/` are unchanged.

- **Redrawn charts** (`05_model_agreement.png`, `08_elo_intervals.png`): the same data as figures 21 and 43, in the deck's colours, with larger labels and a single panel each.
- **Native charts** (slides 2, 3, 4, 7, 9, 10): drawn by PowerPoint from `source/chart_data.json`, so they stay sharp and editable.
- **App screenshots** (`01_*`, `11_*`): new captures of the running app at twice the resolution of `docs/screenshots/`. They show the Atalanta vs Bologna fixture on 7 November 2026, with data up to 2 November 2026.

Colours are the same throughout, and match the app:
- **Outcomes:** home win blue, draw grey, away win orange.
- **Models:** Logistic Regression green, Random Forest purple, XGBoost pink.
- **Bookmaker:** gold.

## Where the numbers come from

Every number is read from files that already exist:
- the processed data;
- the saved experiment results in `outputs/`;
- the published tables in [methodology.md](../methodology.md).

Nothing is retrained or rerun. The Elo intervals on slide 8 are recomputed from the saved Elo predictions with the same fixed-seed bootstrap as notebook 08, and match the published values (for example, Random Forest's corrected interval ends at +0.0001).

Three derived numbers are worth knowing:
- **Slide 7:** "improvement over the baseline" is the baseline's pooled log loss (1.092) minus each forecast's.
- **Slide 9:** "gain over a coin flip" is log 2 (0.693) minus each method's pooled Over 2.5 log loss.
- **Both charts** start at zero, so bar lengths are not exaggerated.

## Rebuilding the deck

From the project folder, after the data, models and experiment results exist (see the main [README](../../README.md#reproducing-the-research)):

```powershell
python docs/presentation/source/prepare_figures.py   # figures/ and source/chart_data.json
npm install pptxgenjs                                 # once, in any folder
node docs/presentation/source/build_deck.js           # set NODE_PATH to that folder's node_modules if needed
```

The app screenshots are not rebuilt by these scripts. They were captured from `streamlit run app.py` in a headless browser at device scale factor 2, and cropped.

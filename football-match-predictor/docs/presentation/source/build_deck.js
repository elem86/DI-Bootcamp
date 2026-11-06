// Builds docs/presentation/serie_a_match_predictor.pptx (12 slides, 16:9, speaker notes on every slide).
//
// Run from the project folder, after prepare_figures.py:
//     npm install pptxgenjs            (anywhere; point NODE_PATH at its node_modules if needed)
//     node docs/presentation/source/build_deck.js
//
// Every number comes from chart_data.json (written by prepare_figures.py from the saved results)
// or from the published results in docs/methodology.md. Nothing here is estimated or illustrative.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const ROOT = path.resolve(HERE, "..");
const FIG = (name) => path.join(ROOT, "figures", name);
const OUT = path.join(ROOT, "serie_a_match_predictor.pptx");
const D = JSON.parse(fs.readFileSync(path.join(HERE, "chart_data.json"), "utf8"));

// ---------------------------------------------------------------- design system
const NAVY = "0B1F3A", NAVY2 = "15305A", INK = "1B2433", MUTED = "5B6577", FAINT = "8A93A3";
const SURFACE = "F3F5F8", GRID = "E3E6EB", WHITE = "FFFFFF", ICE = "9CC3F5";
const HOME = "2A78D6", DRAW = "9AA1AB", AWAY = "EB6834";
const LR = "1BAF7A", RF = "4A3AA7", XGB = "C44E9B", BOOK = "C9971C";
const HEAD = "Arial", BODY = "Calibri";
const X0 = 0.7, W = 11.93; // content margins on a 13.333" x 7.5" slide

const THEME = {
  name: "Serie A Match Predictor", headFontFace: HEAD, bodyFontFace: BODY,
  colors: { dk1: INK, lt1: WHITE, dk2: NAVY, lt2: SURFACE, accent1: HOME, accent2: AWAY, accent3: DRAW,
            accent4: LR, accent5: RF, accent6: XGB, hlink: HOME, folHlink: RF },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Serie A Match Predictor";
pres.subject = "Predicting Serie A results from historical data: a data science case study";
pres.author = "Serie A Match Predictor project";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

function defineLayout(name, dark, titleBox = { x: X0, y: 0.78, w: W, h: 0.8, fontSize: 30 }) {
  pres.defineSlideMaster({
    title: name,
    background: { color: dark ? NAVY : WHITE },
    objects: [
      { text: { text: "Serie A Match Predictor", options: { x: X0, y: 7.0, w: 5, h: 0.3, fontFace: BODY, fontSize: 10,
                                                             color: dark ? "8FA3C0" : FAINT, margin: 0 } } },
      { placeholder: { options: { name: "kicker", type: "body", x: X0, y: 0.42, w: W, h: 0.32, fontFace: HEAD,
                                  fontSize: 12, bold: true, color: dark ? ICE : HOME, margin: 0, valign: "top", align: "left" },
                       text: "" } },
      { placeholder: { options: { name: "title", type: "title", ...titleBox, fontFace: HEAD,
                                  bold: true, color: dark ? WHITE : NAVY, margin: 0, valign: "top", align: "left" },
                       text: "" } },
    ],
    slideNumber: { x: 12.03, y: 7.0, w: 0.6, h: 0.3, fontFace: BODY, fontSize: 10, color: dark ? "8FA3C0" : FAINT,
                   align: "right" },
  });
}
defineLayout("LIGHT", false);
defineLayout("DARK", true);
defineLayout("TITLE", true, { x: X0, y: 1.15, w: 6.3, h: 2.5, fontSize: 40 });

const sections = new Set();
function newSlide(layout, section, kicker, title) {
  if (!sections.has(section)) {
    pres.addSection({ title: section });
    sections.add(section);
  }
  const slide = pres.addSlide({ masterName: layout, sectionTitle: section });
  slide.addText(kicker, { placeholder: "kicker" });
  slide.addText(title, { placeholder: "title" });
  return slide;
}

// ---------------------------------------------------------------- helpers
const pngSize = (file) => {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
};

function image(slide, name, x, y, maxW, maxH, opts = {}) {
  const file = FIG(name);
  const { w, h } = pngSize(file);
  const scale = Math.min(maxW / w, maxH / h);
  const iw = w * scale, ih = h * scale;
  const ix = opts.align === "center" ? x + (maxW - iw) / 2 : x;
  slide.addImage({ path: file, x: ix, y, w: iw, h: ih, altText: opts.alt || name,
                   shadow: opts.shadow ? { type: "outer", color: "000000", opacity: 0.18, blur: 8, offset: 3, angle: 90 }
                                       : undefined });
  return { x: ix, y, w: iw, h: ih };
}

function card(slide, x, y, w, h, fill = SURFACE, name = "card") {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 },
                                                  rectRadius: 0.12, objectName: name });
}

function text(slide, value, x, y, w, h, opts = {}) {
  slide.addText(value, { x, y, w, h, isTextBox: true, fontFace: BODY, fontSize: 16, color: INK, margin: 0,
                         valign: "top", ...opts });
}

function stat(slide, x, y, w, number, label, color, opts = {}) {
  text(slide, number, x, y, w, 0.8, { fontFace: HEAD, fontSize: opts.size || 40, bold: true, color });
  text(slide, label, x, y + (opts.gap || 0.82), w, opts.labelH || 0.75, { fontSize: 14,
                                                                          color: opts.labelColor || MUTED });
}

function bullets(slide, items, x, y, w, h, opts = {}) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 16 }, breakLine: i < items.length - 1,
                                                             paraSpaceAfter: 8 } })),
                { x, y, w, h, isTextBox: true, fontFace: BODY, fontSize: opts.size || 16, color: opts.color || INK,
                  margin: 0, valign: "top" });
}

const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const CHART_TEXT = { catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY, dataLabelFontFace: BODY,
                     legendFontFace: BODY, titleFontFace: BODY, catAxisTitleFontFace: BODY,
                     valAxisTitleFontFace: BODY, catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
                     catAxisLabelFontSize: 13, valAxisLabelFontSize: 12, dataLabelColor: INK,
                     valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" },
                     catAxisLineShow: true, catAxisLineColor: "C9CED6", valAxisLineShow: false };

// ---------------------------------------------------------------- slide 1: the challenge
{
  const s = newSlide("TITLE", "Introduction", "DATA SCIENCE CASE STUDY",
                     "Can 30 years of results predict a football match?");
  text(s, "Three machine-learning models turn past Serie A results into home, draw and away probabilities, "
         + "tested honestly against the bookmakers.", X0, 3.75, 6.1, 1.0, { fontSize: 18, color: "C7D3E6" });
  const stats = [["34", "seasons, 1993-94 to 2026-27"], ["11,826", "completed matches"],
                 ["3", "models, plus a bookmaker benchmark"]];
  stats.forEach(([n, l], i) => stat(s, X0 + i * 2.1, 5.0, 1.95, n, l, WHITE, { size: 34, labelColor: "AFC0D8",
                                                                             gap: 0.62, labelH: 0.6 }));
  image(s, "01_app_predictions_hero.png", 7.35, 0.75, 5.3, 5.85, { align: "center", shadow: true,
        alt: "Match predictions tab of the app: three models' probabilities for Atalanta vs Bologna" });
  text(s, "Data: Football-Data.co.uk, results to 2 Nov 2026", 7.35, 6.68, 5.3, 0.25,
       { fontSize: 10, color: "8FA3C0", align: "right" });
  s.addNotes(
`Here's where we ended up: an app where you pick two Serie A clubs and a date, and three machine-learning models each give you the probability of a home win, a draw and an away win.

Today I want to tell you how we got there, and more importantly how we checked whether these numbers deserve any trust.

The data is 34 seasons of Serie A, from 1993-94 to the current season: 11,826 matches, free from Football-Data.co.uk.

One thing to stress from the start: we're not trying to "pick winners". We want probabilities. If the model says 48% home win, home teams in that situation should win about 48% of the time. Most of this talk is about how you test that.

And a spoiler: some of the most interesting results are things that did not work, plus an honest comparison with the bookmakers.

Transition: before building anything, we needed to understand what we were trying to predict.`);
}

// ---------------------------------------------------------------- slide 2: results
{
  const s = newSlide("LIGHT", "The data", "01  UNDERSTANDING SERIE A RESULTS", "Home wins 45% of the time: the bar to beat");
  const o = D.outcome_shares;
  text(s, `Share of all ${o.matches.toLocaleString("en-US")} matches, 1993-94 to 2 Nov 2026`, X0, 1.85, 7.4, 0.35,
       { fontSize: 14, color: MUTED });
  s.addChart(pres.charts.BAR, [{ name: "Share of matches", labels: ["Home win", "Draw", "Away win"],
                                  values: [o.home, o.draw, o.away] }],
             { x: X0 - 0.1, y: 2.2, w: 7.6, h: 4.4, barDir: "bar", chartColors: [HOME, DRAW, AWAY],
               catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1,
               valAxisLabelFormatCode: "0%", showValue: true, dataLabelPosition: "outEnd",
               dataLabelFormatCode: "0.0%", dataLabelFontSize: 18, dataLabelFontBold: true, barGapWidthPct: 55,
               showLegend: false, ...CHART_TEXT, catAxisLabelFontSize: 18, objectName: "outcome shares chart" });
  card(s, 8.75, 1.85, 3.88, 2.25);
  stat(s, 9.05, 2.05, 3.4, pct(o.home), "of all matches are home wins. A forecast that always says “home win” is right this often.", HOME,
       { labelH: 1.1 });
  card(s, 8.75, 4.3, 3.88, 2.25);
  stat(s, 9.05, 4.5, 3.4, o.goals_per_match.toFixed(2), "goals per match. The most common scores are 1-1, 1-0 and 2-1: one goal often decides it.", INK,
       { labelH: 1.1 });
  s.addNotes(
`Let's start with what we're predicting. Of all 11,826 matches, 45.4% were home wins, and draws and away wins were 27.3% each.

So a "model" that ignores everything and always says home win would be right about 45% of the time. That's the bar any real model has to clear. (In the seven recent seasons we test on later, home wins were rarer, so that naive pick is right only 41% of the time there.)

The second number: 2.66 goals per match, with 1-1, 1-0 and 2-1 the most common scores. When a single goal decides so many matches, luck plays a huge part, so nobody, bookmakers included, is right most of the time.

Notice draws: as common as away wins, but with no obvious favourite. Keep that in mind; draws turn out to be the hardest outcome to call.

Main point: the baseline is about 45%, and football is noisy.

Transition: but these averages hide something important: the game has been changing.`);
}

// ---------------------------------------------------------------- slide 3: change over time
{
  const s = newSlide("LIGHT", "The data", "02  HOW FOOTBALL HAS CHANGED", "Home advantage has been shrinking for 30 years");
  const t = D.shares_by_season;
  text(s, "Share of matches won by the home team, drawn, or won by the away team, each completed season", X0, 1.85, 8.4,
       0.35, { fontSize: 14, color: MUTED });
  s.addChart(pres.charts.LINE, [
      { name: "Home win", labels: t.seasons, values: t.home },
      { name: "Draw", labels: t.seasons, values: t.draw },
      { name: "Away win", labels: t.seasons, values: t.away }],
    { x: X0 - 0.1, y: 2.2, w: 8.6, h: 4.5, chartColors: [HOME, DRAW, AWAY], lineSize: 3, lineDataSymbol: "none",
      valAxisMinVal: 0, valAxisMaxVal: 0.6, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0%",
      catAxisLabelFrequency: 4, showLegend: true, legendPos: "t", legendFontSize: 14, ...CHART_TEXT,
      objectName: "outcome shares by season chart" });
  card(s, 9.2, 2.2, 3.43, 2.0);
  stat(s, 9.45, 2.4, 3.0, "50% → 40%", "home wins: 1990s average vs the last five seasons", HOME, { size: 32, gap: 0.7 });
  card(s, 9.2, 4.45, 3.43, 2.0);
  stat(s, 9.45, 4.65, 3.0, "21% → 32%", "away wins over the same periods: from 1 in 5 matches to 1 in 3", AWAY, { size: 32, gap: 0.7 });
  s.addNotes(
`This is every completed season from 1993-94 to 2025-26. The blue line is home wins, grey is draws, orange is away wins.

In the 1990s home teams won about half their matches, 49.6% on average. Over the last five seasons it's 40.4%. Away wins went the other way, from about 21% to about 32%: from one match in five to one in three.

You might think of the empty stadiums of 2020-21. Home advantage was low then, but the decline started long before and it didn't come back when fans returned.

At the same time the league became more unequal: the gap between strong and weak clubs grew. So who is playing matters more now than where.

Two lessons for modelling. Team-strength features should matter most. And evaluation has to respect time: if we shuffled all the matches and tested on a random sample, the model would learn from the future and be tested on the past.

Main point: the past is not a perfect guide to the present, so always test on later seasons.

Transition: so what can a model actually know about a match before kick-off?`);
}

// ---------------------------------------------------------------- slide 4: leakage
{
  const s = newSlide("LIGHT", "Method", "03  LEAKAGE-SAFE FEATURE ENGINEERING", "No peeking: every feature comes from the past");
  // match-day timeline (editable shapes)
  const ty = 2.55;
  s.addShape(pres.shapes.LINE, { x: X0 + 0.2, y: ty, w: W - 0.4, h: 0, line: { color: "C9CED6", width: 2 } });
  const dayX = 7.6;
  for (let i = 0; i < 9; i++) {
    s.addShape(pres.shapes.OVAL, { x: X0 + 0.45 + i * 0.68, y: ty - 0.14, w: 0.28, h: 0.28,
                                   fill: { color: HOME }, line: { color: WHITE, width: 1.5 }, objectName: "earlier match" });
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: dayX - 0.55, y: ty - 0.34, w: 1.1, h: 0.68, fill: { color: NAVY },
                                              line: { color: NAVY }, rectRadius: 0.1, objectName: "fixture" });
  text(s, "Fixture", dayX - 0.55, ty - 0.34, 1.1, 0.68, { fontSize: 14, bold: true, color: WHITE, align: "center",
                                                          valign: "middle" });
  [dayX + 0.75, dayX + 1.15].forEach((x) => s.addShape(pres.shapes.OVAL, { x, y: ty - 0.14, w: 0.28, h: 0.28,
      fill: { color: WHITE }, line: { color: DRAW, width: 1.5, dashType: "dash" }, objectName: "same-day match" }));
  for (let i = 0; i < 4; i++) {
    s.addShape(pres.shapes.OVAL, { x: 9.7 + i * 0.68, y: ty - 0.14, w: 0.28, h: 0.28, fill: { color: GRID },
                                   line: { color: GRID }, objectName: "later match" });
  }
  text(s, "Earlier matches: used", X0 + 0.45, ty + 0.35, 5.5, 0.35, { fontSize: 15, bold: true, color: HOME });
  text(s, "Same day: not used", dayX - 0.2, ty + 0.35, 2.1, 0.35, { fontSize: 15, bold: true, color: MUTED });
  text(s, "Later results: never used", 9.7, ty + 0.35, 2.93, 0.35, { fontSize: 15, bold: true, color: FAINT });
  text(s, "Match day", dayX - 0.55, ty - 0.75, 1.1, 0.3, { fontSize: 12, color: MUTED, align: "center" });

  // form vs result
  const f = D.form_vs_result;
  s.addChart(pres.charts.LINE, [
      { name: "Home win", labels: f.centres, values: f.home },
      { name: "Draw", labels: f.centres, values: f.draw },
      { name: "Away win", labels: f.centres, values: f.away }],
    { x: X0 - 0.1, y: 3.45, w: 7.3, h: 3.35, chartColors: [HOME, DRAW, AWAY], lineSize: 3, lineDataSymbol: "circle",
      lineDataSymbolSize: 6, valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2,
      valAxisLabelFormatCode: "0%", showLegend: true, legendPos: "t", legendFontSize: 13, showCatAxisTitle: true,
      catAxisTitle: "Home minus away points per game, last 10 matches", catAxisTitleFontSize: 12,
      catAxisTitleColor: MUTED, ...CHART_TEXT, catAxisLabelFontSize: 11, objectName: "form vs result chart" });
  card(s, 8.15, 3.55, 4.48, 3.15);
  bullets(s, ["25 numbers per club: recent form, home or away form, this season, last season, promoted",
              "Built only from matches on earlier dates",
              "Tested: changing later results never changes earlier features"], 8.4, 3.8, 4.0, 2.8, { size: 16 });
  s.addNotes(
`Models can't read a fixture list; they need numbers. For every match we built 50 features, 25 per club, describing how each team looked before kick-off: recent form, home or away form, this season so far, last season, and whether the club was just promoted.

The chart shows they work. Move right and the home team is in much better form: home wins climb from about 15% to about 75%. But the grey draw line stays between roughly 19% and 31% whatever the gap.

The key idea here is data leakage: a model accidentally seeing information it wouldn't have in real life. If "goal difference this season" included the match's own score, the model would be reading the answer: brilliant in testing, useless in reality.

Our rule, shown in the timeline: only matches on earlier dates, not even other games that day. Automated tests rewrite later results and check that nothing earlier changes.

Transition: with these features, we trained three very different kinds of model.`);
}

// ---------------------------------------------------------------- slide 5: three models
{
  const s = newSlide("LIGHT", "Method", "04  THREE MACHINE-LEARNING MODELS", "Three algorithms, almost the same answer");
  const a = D.model_agreement;
  image(s, "05_model_agreement.png", X0, 1.8, 5.4, 4.95, { alt: "Scatter of XGBoost against Logistic Regression home-win probabilities, 2025-26" });
  const models = [["Logistic Regression", "Adds up the evidence for each outcome: a linear model", LR],
                  ["Random Forest", "Hundreds of decision trees that vote", RF],
                  ["XGBoost", "Trees built one after another, each correcting the last", XGB]];
  models.forEach(([name, desc, color], i) => {
    const y = 1.85 + i * 1.12;
    card(s, 6.55, y, 6.08, 0.95);
    s.addShape(pres.shapes.OVAL, { x: 6.8, y: y + 0.32, w: 0.3, h: 0.3, fill: { color }, line: { color } });
    text(s, name, 7.3, y + 0.14, 5.1, 0.35, { fontFace: HEAD, fontSize: 17, bold: true, color: INK });
    text(s, desc, 7.3, y + 0.5, 5.1, 0.35, { fontSize: 14, color: MUTED });
  });
  stat(s, 6.55, 5.3, 2.9, `${a.corr_rf_lr.toFixed(2)}–${a.corr_xgb_lr.toFixed(2)}`,
       "correlation between their home-win probabilities", INK, { size: 34, gap: 0.62, labelH: 0.7 });
  stat(s, 9.7, 5.3, 2.93, pct(a.same_pick_all_three, 0), "of 2025-26 matches: all three pick the same outcome", INK,
       { size: 34, gap: 0.62, labelH: 0.7 });
  s.addNotes(
`We trained three well-known classifiers on exactly the same features. Logistic Regression is a linear model: it adds up evidence for each outcome. Random Forest is hundreds of decision trees voting. XGBoost builds trees one after another, each fixing the last one's mistakes.

The chart is the surprise. Each dot is one match from 2025-26; across is Logistic Regression's home-win probability, up is XGBoost's. If they disagreed, this would be a wide cloud. Instead it hugs the diagonal: correlation 0.98, and 0.97 for Random Forest. All three pick the same outcome in 88% of matches.

A quick word on overfitting: a flexible model can memorise noise, like a student memorising last year's exam answers. With noisy football data, the settings that won were the simple ones: strong regularisation, and XGBoost trees only two levels deep.

One honesty note: 2025-26 is the season we used to choose each model's settings, so it's a selection season, not an independent test. The next slide fixes that.

Main point: the algorithm is not what limits us. Better predictions need better information, not a fancier model.

Transition: so how do we judge them fairly?`);
}

// ---------------------------------------------------------------- slide 6: walk-forward
{
  const s = newSlide("LIGHT", "Method", "05  CHRONOLOGICAL WALK-FORWARD EVALUATION", "Testing like a forecaster, one season at a time");
  const cols = ["1993-94\nto 2015-16", "16-17", "17-18", "18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25",
                "25-26", "26-27"];
  const labelW = 1.55, gx = X0 + labelW, firstW = 1.0, cw = 0.48, gap = 0.04, rh = 0.33, rgap = 0.07;
  const colX = (j) => (j === 0 ? gx : gx + firstW + gap + (j - 1) * (cw + gap));
  const colW = (j) => (j === 0 ? firstW : cw);
  const top = 2.25;
  cols.forEach((c, j) => text(s, c, colX(j), 1.78, colW(j), 0.45, { fontSize: 11, color: MUTED, align: "center",
                                                                     valign: "bottom" }));
  const TRAIN = "C9D6EA", CHOOSE = "7FA6DD", PREDICT = NAVY;
  const cell = (j, y, kind) => {
    const opts = { x: colX(j), y, w: colW(j), h: rh, rectRadius: 0.05, objectName: kind };
    if (kind === "train") Object.assign(opts, { fill: { color: TRAIN }, line: { color: TRAIN } });
    if (kind === "choose") Object.assign(opts, { fill: { color: CHOOSE }, line: { color: CHOOSE } });
    if (kind === "predict") Object.assign(opts, { fill: { color: PREDICT }, line: { color: PREDICT } });
    if (kind === "hidden") Object.assign(opts, { fill: { color: WHITE }, line: { color: "D5DAE1", width: 0.75 } });
    if (kind === "test") Object.assign(opts, { fill: { color: WHITE }, line: { color: NAVY, width: 1.75 } });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, opts);
  };
  for (let k = 0; k < 7; k++) {
    const season = 2018 + k, y = top + k * (rh + rgap);
    text(s, `Predict ${season}-${String(season + 1).slice(2)}`, X0, y, labelW - 0.1, rh,
         { fontSize: 12, color: INK, valign: "middle" });
    cols.forEach((_, j) => {
      const yr = j === 0 ? 2015 : 2015 + j;
      cell(j, y, yr <= season - 2 ? "train" : yr === season - 1 ? "choose" : yr === season ? "predict" : "hidden");
    });
  }
  const yApp = top + 7 * (rh + rgap) + 0.25;
  text(s, "Saved models\n(used by the app)", X0, yApp - 0.08, labelW - 0.05, 0.5, { fontSize: 12, bold: true,
                                                                                    color: INK, valign: "middle" });
  cols.forEach((_, j) => cell(j, yApp, j <= 9 ? "train" : j === 10 ? "choose" : "test"));
  // legend
  const legend = [["Train", TRAIN, null], ["Choose settings", CHOOSE, null], ["Predict and score", PREDICT, null],
                  ["Hidden", WHITE, "D5DAE1"], ["First 100 matches of 2026-27: held out at first, examined since", WHITE, NAVY]];
  let lx = X0, ly = yApp + 0.6;
  legend.forEach(([label, fill, border], i) => {
    if (i === 4) { lx = X0; ly += 0.38; }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y: ly + 0.04, w: 0.3, h: 0.22, rectRadius: 0.04,
      fill: { color: fill }, line: { color: border || fill, width: border === NAVY ? 1.75 : 0.75 } });
    const lw = i === 4 ? 6.0 : [0.7, 1.6, 1.7, 0.9][i];
    text(s, label, lx + 0.38, ly, lw, 0.3, { fontSize: 12, color: MUTED, valign: "middle" });
    lx += 0.38 + lw + 0.25;
  });
  // log loss card
  card(s, 9.55, 1.85, 3.08, 4.85);
  text(s, "How we score", 9.8, 2.05, 2.6, 0.4, { fontFace: HEAD, fontSize: 18, bold: true, color: NAVY });
  text(s, "Log loss: the probability you gave to what actually happened, as a penalty. Lower is better.", 9.8, 2.5,
       2.6, 1.05, { fontSize: 14, color: MUTED });
  const ex = [["Said 50%, it happened", "0.69"], ["Said 10%, it happened", "2.30"], ["1/3 on every outcome", "1.10"]];
  ex.forEach(([l, v], i) => {
    const y = 3.7 + i * 0.85;
    text(s, l, 9.8, y, 2.6, 0.3, { fontSize: 14, color: INK });
    text(s, v, 9.8, y + 0.3, 2.6, 0.45, { fontFace: HEAD, fontSize: 22, bold: true, color: NAVY });
  });
  s.addNotes(
`Two ideas on this slide.

First, walk-forward testing. Each row is one season we replayed, from 2018-19 to 2024-25. For each one, we hid everything after it, chose the model's settings using only earlier seasons, trained, and predicted. So every prediction used only what was known at the time: seven seasons, 2,657 matches, every forecast scored on the same matches.

The bottom row is the saved models in the app: settings chosen on 2025-26, then refitted. The first 100 matches of 2026-27 were held out at first, but we've looked at those results since, so they can't fairly judge a new model.

Second, scoring. Log loss looks at the probability you gave to what actually happened. Say 50% and it happens: 0.69. Say 10% and it happens anyway: 2.30, a big penalty for overconfidence. Lower is better. We also check calibration, the weather-forecaster test: when you say 40%, it should happen about 40% of the time.

Honesty note: these seven seasons were reused by every later experiment, so later gains on them are exploratory, not proof.

Transition: now the real test: the people who set the odds.`);
}

// ---------------------------------------------------------------- slide 7: bookmaker
{
  const s = newSlide("LIGHT", "Results", "06  RESULTS VERSUS THE BOOKMAKER", "The bookmaker still beats the models");
  const b = D.backtest.forecasts;
  const order = ["Logistic Regression", "Random Forest", "XGBoost", "Bookmaker (Bet365)"];
  text(s, "Log-loss improvement over the baseline, 2018-19 to 2024-25 (higher is better)", X0, 1.85, 7.8, 0.35,
       { fontSize: 14, color: MUTED });
  s.addChart(pres.charts.BAR, [{ name: "Improvement", labels: order, values: order.map((k) => Number(b[k].improvement.toFixed(3))) }],
             { x: X0 - 0.1, y: 2.2, w: 7.9, h: 3.75, barDir: "bar", chartColors: [LR, RF, XGB, BOOK],
               catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 0.14, valAxisMajorUnit: 0.02,
               valAxisLabelFormatCode: "0.00", showValue: true, dataLabelPosition: "outEnd",
               dataLabelFormatCode: "0.000", dataLabelFontSize: 16, dataLabelFontBold: true, barGapWidthPct: 45,
               showLegend: false, ...CHART_TEXT, catAxisLabelFontSize: 15, objectName: "improvement over baseline chart" });
  text(s, `Baseline: the same historical outcome frequencies for every match (log loss ${b.Baseline.log_loss.toFixed(3)}); `
         + "its single pick is always a home win. Scale starts at zero.", X0, 6.05, 7.7, 0.6, { fontSize: 12, color: MUTED });
  card(s, 8.95, 1.85, 3.68, 2.25);
  stat(s, 9.2, 2.05, 3.2, `${b["Bookmaker (Bet365)"].seasons_best} of 7`, "seasons in which the bookmaker had the best log loss", BOOK,
       { labelH: 0.9 });
  card(s, 8.95, 4.3, 3.68, 2.4);
  stat(s, 9.2, 4.5, 3.2, "0.021–0.023", "bookmaker's lead over each model; 95% intervals exclude zero. It may reflect team news the models lack.",
       INK, { size: 32, gap: 0.65, labelH: 1.3 });
  s.addNotes(
`Seven backtest seasons, pooled. The bars show how much each forecast improved on the baseline. Higher is better, and the scale starts at zero.

The baseline gives every match the same probabilities: how often home wins, draws and away wins happened in the training seasons. Its log loss is 1.092. Because home wins are the most common result, its single pick is always "home win", which was right 41% of the time in these seasons.

All three models improve on it by about 0.11, in every one of the seven seasons, and they're within 0.003 of each other.

The gold bar is the bookmaker: Bet365's odds turned into probabilities with the margin removed, used only for comparison. Its lead over the models is 0.021 to 0.023, about a fifth of the distance the models travelled from the baseline, and the 95% intervals exclude zero. It was best in 6 of 7 seasons.

Why? We can't prove it, but the likely reason is information the models don't have: injuries, line-ups, transfers.

Two things we don't claim. The models scored slightly better on the first 100 matches of 2026-27, 0.882 against 0.898, but 100 matches can't separate them. And draws: a draw is rarely anyone's single most likely outcome, the bookmaker's included. They're hard to pick out, not impossible to model.

Transition: could better team-strength measures close the gap?`);
}

// ---------------------------------------------------------------- slide 8: Elo and recency
{
  const s = newSlide("LIGHT", "Experiments", "07  ELO AND RECENCY EXPERIMENTS", "Two good ideas that didn't make the cut");
  text(s, "Adding Elo ratings: change in pooled log loss over 7 seasons", X0, 1.85, 7.8, 0.35, { fontSize: 14, color: MUTED });
  image(s, "08_elo_intervals.png", X0, 2.25, 7.75, 3.9, { alt: "Elo minus control: pooled log-loss change with 95% and corrected intervals for three models" });
  text(s, "Every model improved slightly; after correcting for testing three models, no interval excludes zero.",
       X0, 6.2, 7.75, 0.55, { fontSize: 14, color: INK });
  card(s, 8.95, 1.85, 3.68, 4.85);
  text(s, "Recency weighting", 9.2, 2.05, 3.2, 0.4, { fontFace: HEAD, fontSize: 18, bold: true, color: NAVY });
  text(s, "Six ways to make recent seasons count more", 9.2, 2.5, 3.2, 0.6, { fontSize: 14, color: MUTED });
  stat(s, 9.2, 3.15, 3.2, "−0.0027", "best strategy (5-year half-life), better in 6 of 7 seasons", INK,
       { size: 32, gap: 0.62, labelH: 0.7 });
  text(s, "But its intervals include zero, and it was the best of six tries on the same seasons.", 9.2, 4.6, 3.2,
       1.0, { fontSize: 14, color: INK });
  text(s, "Decision for both: not deployed", 9.2, 5.85, 3.2, 0.6, { fontSize: 15, bold: true, color: AWAY });
  s.addNotes(
`Two ideas that sounded promising.

Elo ratings come from chess: a club gains points by beating others, more for a surprise win. We added leakage-safe Elo ratings to the features and reran the seven-season backtest.

Left of zero means Elo helped, and every model moved slightly left, by 0.0014 to 0.0039. Sounds convincing. But the rule we wrote beforehand required the gain to stay clear of zero after correcting for testing three models: that's the thin line. Random Forest's ordinary 95% interval excludes zero, but its corrected interval ends at plus 0.0001. A miss by a hair, but a miss. The reason: Elo mostly repeats last season's points gap, which the models already have.

Recency: since the game changes, maybe recent seasons deserve more weight. Of six strategies, a 5-year half-life did best, better in 6 of 7 seasons by 0.0027. But its intervals include zero, and it was the best of six tries on the same seasons. That's exactly how luck can look like a discovery.

Neither gain is established, so neither is deployed. Both stay as candidates to test on seasons they haven't seen.

Transition: so 1X2 probabilities are close to what our data allows. What about goals?`);
}

// ---------------------------------------------------------------- slide 9: goals
{
  const s = newSlide("LIGHT", "Experiments", "08  EXPECTED GOALS VERSUS EVENT PROBABILITIES", "We can estimate how many goals, not whether");
  const g = D.expected_goals, o = D.over_2_5;
  card(s, X0, 1.85, 3.75, 4.85);
  text(s, "Expected goals", X0 + 0.25, 2.05, 3.25, 0.4, { fontFace: HEAD, fontSize: 18, bold: true, color: NAVY });
  stat(s, X0 + 0.25, 2.6, 3.25, `${g.improvement_pct.toFixed(1)}%`, "better than the league average (Poisson deviance), in all 7 seasons", LR,
       { size: 44, gap: 0.85, labelH: 0.9 });
  text(s, "Over/Under and both-teams-to-score probabilities: no demonstrated skill, and too extreme. None is shown as a forecast.",
       X0 + 0.25, 4.6, 3.25, 1.6, { fontSize: 15, color: INK });
  const order = ["Market average", "Poisson regression", "Team attack/defence", "League average"];
  const labels = ["Market odds", "Poisson regression", "Team attack/defence", "League average"];
  text(s, "Over 2.5 goals: log-loss gain over a 50/50 coin flip (higher is better)", 4.85, 1.85, 7.8, 0.35,
       { fontSize: 14, color: MUTED });
  s.addChart(pres.charts.BAR, [{ name: "Gain over a coin flip", labels,
                                  values: order.map((k) => Number(o.gain_over_coin_flip[k].toFixed(4))) }],
             { x: 4.75, y: 2.2, w: 7.95, h: 3.8, barDir: "bar", chartColors: [BOOK, "3B5B8C", "7C8BA1", "B0B8C4"],
               catAxisOrientation: "maxMin", valAxisMinVal: -0.008, valAxisMaxVal: 0.02, valAxisMajorUnit: 0.004,
               valAxisLabelFormatCode: "0.000", showValue: true, dataLabelPosition: "outEnd",
               dataLabelFormatCode: "+0.0000;−0.0000", dataLabelFontSize: 15, dataLabelFontBold: true,
               barGapWidthPct: 45, showLegend: false, ...CHART_TEXT, catAxisLabelFontSize: 14,
               catAxisLabelPos: "low", objectName: "over 2.5 gain over coin flip chart" });
  text(s, `A coin flip scores ${o.coin_flip.toFixed(3)}. ${o.matches.toLocaleString("en-US")} matches, 2018-19 to 2024-25 `
         + `(market odds: ${o.market_matches.toLocaleString("en-US")}). Zero = no better than a coin flip.`,
       4.85, 6.1, 7.8, 0.6, { fontSize: 12, color: MUTED });
  s.addNotes(
`Many people care less about who wins than about goals: will there be more than 2.5 goals, will both teams score?

Goals in football follow a classic statistical pattern, the Poisson distribution, so you can estimate each side's expected goals and derive any goal probability from them. On expected goals, our Poisson regression is clearly good: 10.3% better than the league average, in all seven seasons.

But turn that into a yes-or-no probability and it falls apart. The chart shows how much each method beats a 50/50 coin flip on "over 2.5 goals". Zero means no better than a coin flip. Our best model gains 0.002. The league average is actually slightly worse than a coin flip. The market's Over/Under odds gain 0.015, about seven times our best.

The probabilities were also too extreme: when the model said 70%, it happened less often than that. And it under-predicted both-teams-to-score by about 4 percentage points.

So estimating the average number of goals is not the same as getting yes/no probabilities right. These models failed their readiness rule, and the app shows no goal forecasts. Its goals tab shows only history, labelled "not a forecast".

Transition: we went looking for why, and found it in the league itself.`);
}

// ---------------------------------------------------------------- slide 10: adaptive rates
{
  const s = newSlide("LIGHT", "Experiments", "09  ADAPTIVE SCORING RATES", "Scoring drifts, and following it almost works");
  const g = D.goals_by_season;
  const last = g.seasons.indexOf("2024-25") + 1;
  text(s, "Goals per match by season, 1993-94 to 2024-25 (the seasons the experiments could use)", X0, 1.85, 7.9, 0.35,
       { fontSize: 14, color: MUTED });
  s.addChart(pres.charts.LINE, [
      { name: "Home team goals", labels: g.seasons.slice(0, last), values: g.home.slice(0, last) },
      { name: "Away team goals", labels: g.seasons.slice(0, last), values: g.away.slice(0, last) }],
    { x: X0 - 0.1, y: 2.2, w: 8.0, h: 4.5, chartColors: [HOME, AWAY], lineSize: 3, lineDataSymbol: "none",
      valAxisMinVal: 0, valAxisMaxVal: 2.0, valAxisMajorUnit: 0.5, valAxisLabelFormatCode: "0.0",
      catAxisLabelFrequency: 4, showLegend: true, legendPos: "t", legendFontSize: 14, ...CHART_TEXT,
      objectName: "goals per match by season chart" });
  card(s, 8.6, 1.85, 4.03, 4.85);
  text(s, "Following the last 200 matches", 8.85, 2.05, 3.6, 0.7, { fontFace: HEAD, fontSize: 17, bold: true, color: NAVY });
  text(s, "Over 2.5 goals, adaptive frequency vs the five rules set in advance", 8.85, 2.75, 3.6, 0.6,
       { fontSize: 13, color: MUTED });
  const checks = [["✓", "Clear gain over the static frequency", LR], ["✓", "Not worse than the best model", LR],
                  ["✓", "Better in 6 of 7 seasons", LR], ["✓", "Survives leaving out any season", LR],
                  ["✗", "Calibration: adjusts too little", AWAY]];
  checks.forEach(([mark, label, color], i) => {
    const y = 3.4 + i * 0.5;
    text(s, mark, 8.85, y, 0.35, 0.4, { fontFace: "Segoe UI Symbol", fontSize: 18, bold: true, color });
    text(s, label, 9.25, y + 0.03, 3.25, 0.4, { fontSize: 15, color: INK });
  });
  text(s, "Promising, not validated: not deployed", 8.85, 5.95, 3.6, 0.5, { fontSize: 15, bold: true, color: AWAY });
  s.addNotes(
`Home goals in blue, away goals in orange, season by season.

On the right, total goals went from 2.68 in 2018-19 to just over 3.0 in the next two seasons, then back to about 2.6. Away scoring has stayed well above its thirty-year average. A forecaster anchored to the long-run average is off in a consistent direction: that's drift.

So we followed the league: estimate the scoring level from the last 200 matches, pulled gently toward history, with settings fixed in advance.

It gave the best goal-event scores of any method. For over 2.5 goals it passed four of the five rules written in advance: a clear gain over the static frequency even after correcting for 8 comparisons, not worse than the best model, better in 6 of 7 seasons, and robust to leaving out any season. It failed calibration: right direction, but too timid.

We didn't tweak the settings afterwards, because that would be tuning on the test. And it's still behind the market: 0.6886 against 0.6786 on over 2.5.

So: our most promising goals idea, but exploratory. It needs seasons it has never seen.

Transition: all of this evidence ends up in the app.`);
}

// ---------------------------------------------------------------- slide 11: the app
{
  const s = newSlide("LIGHT", "Application", "10  THE STREAMLIT APPLICATION", "An app that is honest about what it knows");
  const shots = [["Match predictions", "model output", "11_app_predictions_tab.png", HOME],
                 ["Goals & BTTS history", "descriptive, not a forecast", "11_app_goals_tab.png", MUTED],
                 ["Research & methodology", "the evidence, including what isn't used", "11_app_research_tab.png", NAVY]];
  const iw = 3.83, gapX = (W - 3 * iw) / 2;
  shots.forEach(([name, what, file, color], i) => {
    const x = X0 + i * (iw + gapX);
    text(s, name, x, 1.85, iw, 0.35, { fontFace: HEAD, fontSize: 16, bold: true, color: INK });
    text(s, what, x, 2.2, iw, 0.3, { fontSize: 13, color });
    s.addShape(pres.shapes.RECTANGLE, { x: x - 0.01, y: 2.6 - 0.01, w: iw + 0.02, h: iw * 1059 / 2012 + 0.02,
                                        fill: { color: WHITE }, line: { color: "D5DAE1", width: 0.75 } });
    image(s, file, x, 2.6, iw, iw * 1059 / 2012, { alt: `${name} tab of the app` });
  });
  const safeguards = [["Freshness", "Shows the latest match date; warns when data is over 10 days old"],
                      ["Honest demos", "Replays past matches as they looked that morning, and flags training seasons"],
                      ["No hidden models", "No unvalidated forecasts; the app never retrains anything"]];
  safeguards.forEach(([head, body], i) => {
    const x = X0 + i * (iw + gapX);
    card(s, x, 4.9, iw, 1.5);
    text(s, head, x + 0.25, 5.1, iw - 0.5, 0.35, { fontFace: HEAD, fontSize: 16, bold: true, color: NAVY });
    text(s, body, x + 0.25, 5.5, iw - 0.5, 0.8, { fontSize: 15, color: INK });
  });
  s.addNotes(
`Everything we learned ends up in a Streamlit app with three tabs that keep three kinds of content apart.

Match predictions: pick two clubs and a date and see each model's probabilities, the fair odds they imply, and where the models disagree. The fixture's 50 features are rebuilt from matches before that date, using the same code as the training data.

Goals and BTTS history: each club's recent goal record. It's clearly labelled as history, not a forecast, because the goal models didn't pass their rules.

Research and methodology: the evidence from this talk, read straight from the saved result files, including why Elo and the goal models aren't used.

And some safeguards. The latest match date is always visible, and a warning appears when the data is more than 10 days old or the fixture is far beyond it. Historical demo mode replays any past match as it looked that morning, and warns when that season was used for training. The app never retrains anything.

Behind it: 436 automated tests, from leakage checks to the app's layout, and rebuilding everything from a fresh copy of the repository reproduced the published results.

Transition: so, what did thirty years of Serie A teach us?`);
}

// ---------------------------------------------------------------- slide 12: conclusions
{
  const s = newSlide("DARK", "Conclusions", "CONCLUSIONS", "What worked, what didn't, what we learned");
  const cols = [
    ["What worked", LR, ["Leakage-safe features, tested", "Models beat the baseline in every season",
                         "A fair, reusable testing framework", "Expected goals"]],
    ["What didn't", AWAY, ["Beating the bookmaker", "Picking draws as the most likely outcome",
                           "Elo and recency: gains not confirmed", "Over/Under and BTTS probabilities"]],
    ["What we learned", ICE, ["The algorithm barely mattered", "Seasons vary more than models",
                              "Accuracy can mislead; score probabilities", "Write the rule before you look"]],
  ];
  const cw = 3.83, gapX = (W - 3 * cw) / 2;
  cols.forEach(([head, color, items], i) => {
    const x = X0 + i * (cw + gapX);
    card(s, x, 1.85, cw, 3.95, NAVY2);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 2.12, w: 0.26, h: 0.26, fill: { color }, line: { color } });
    text(s, head, x + 0.7, 2.03, cw - 0.9, 0.45, { fontFace: HEAD, fontSize: 20, bold: true, color: WHITE });
    bullets(s, items, x + 0.3, 2.75, cw - 0.6, 2.9, { size: 18, color: "DCE5F2" });
  });
  text(s, "Next: test the leading candidates on seasons they have never seen, and add new information such as line-ups.",
       X0, 6.15, W, 0.45, { fontSize: 17, italic: true, color: ICE });
  s.addNotes(
`Three columns to finish.

What worked: a leakage-safe pipeline from thirty years of messy spreadsheets; models that beat the baseline by about 0.11 log loss in every season; a testing framework that makes the next idea cheap to test and hard to fool ourselves with; and good expected-goals estimates.

What didn't: we didn't beat the bookmaker, which stayed about 0.02 ahead. Draws are rarely anyone's most likely outcome. The Elo and recency gains weren't confirmed, and no Over/Under or both-teams-to-score model was good enough to ship.

What we learned: information most likely matters more than algorithms, because three very different models gave almost the same answer. Seasons vary more than models, so one season can flatter or sink any model. Accuracy can mislead, so score the probabilities. And write the rule before you look, so exploratory gains don't turn into false discoveries.

Next: test Random Forest with Elo and the adaptive goal frequency on seasons they've never seen, and add information such as line-ups.

Thank you. Happy to take questions.`);
}

// ---------------------------------------------------------------- write
(async () => {
  await pres.writeFile({ fileName: OUT });
  if (process.env.APPLY_THEME_MODULE) {
    const { applyTheme } = require(process.env.APPLY_THEME_MODULE);
    await applyTheme(OUT, THEME);
  }
  console.log(`Wrote ${OUT}`);
})();

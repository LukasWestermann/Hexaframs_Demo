# hexafarms · Yield Forecast Demo

Interactive strawberry yield forecasting demo built for the hexafarms ML Engineer interview.

## What it does

- **Mechanistic baseline**: flush curve in thermal time (GDD, Tbase 3°C), fitted to synthetic multi-season harvest data
- **Hybrid ML layer**: gradient-boosted residual on top of the physics — toggle in the sidebar
- **Interactive UI**: block configuration, season plan, picking capacity, forecast horizon

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Repo structure

```
app.py          Streamlit UI — charts, KPIs, forward planning table
model.py        Mechanistic baseline + hybrid (HGB residual) model
simulator.py    Synthetic harvest data generator (5 seasons, 4 systems, 3 varieties)
requirements.txt
```

## Key design decisions

**Why mechanistic first, ML second**
At 4 seasons of training data, the physics does the heavy lifting (7.35% MAPE vs 7.05% hybrid on farm-level weekly total — within noise). The ML residual layer earns its keep as data accumulates; it also picks up block quality effects the flush curve can't see. New customers get a working forecast from day one.

**Why thermal time features matter**
Trees extrapolate correctly outside the training envelope of planting dates *only* when they see tau (cumulative GDD) and weekly GDD as features. Without these, a late-planted block at week 34 gets a 114% error. The feature set carries the mechanism.

**What's synthetic**
Everything. Five seasons, 14 blocks/season, Rhineland-like weather. The noise structure (AR-1 weekly wiggle, lognormal block quality, shock weeks) creates a realistic error floor of ~7% MAPE that neither model can beat. Flush intervals, tau0, and sigma are tuned to the published literature but not calibrated to real Hensen data.

**Known upper bound**
The model sees realised weather for the week it predicts — zero weather forecast error. Real 3-week-ahead accuracy will be ~1.5pp worse (weather MAE ≈ 1.9°C at 3 weeks). State this before anyone asks.

## Demo talking points

1. Open sidebar, show the model toggle — "here's what the ML layer adds at 4 seasons of data"
2. Drag the plant week slider for Block C — the flush timing shifts, the cumulative chart adjusts
3. Point at the actuals card — "model within 5% over the past 6 weeks"
4. Push the capacity input down — over-capacity weeks light up amber in the forward table
5. Raise the plan above the forecast total — KPI turns red — "this is the gap the planting decision needs to close"

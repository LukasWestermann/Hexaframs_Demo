"""
hexafarms · Yield Forecast Demo
================================
Run:  streamlit run app.py
"""
import io
from datetime import datetime, time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from simulator import build, weather, block_frame, VARIETIES, SYSTEMS, TBASE
from model import fit_baseline, base_pred, fit_hybrid, make_forecast, week_to_date

st.set_page_config(
    page_title="hexafarms · Yield Forecast",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: .04em; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
.sync-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; color: #5f5e5a;
}
.sync-dot { width: 8px; height: 8px; border-radius: 50%; background: #0ca30c; display: inline-block; }
section[data-testid="stSidebar"] { min-width: 280px !important; }
</style>
""", unsafe_allow_html=True)

CURRENT_WEEK = 26
BLOCK_COLORS = ["#1D9E75", "#7F77DD", "#BA7517", "#D85A30"]

# Hardcoded actuals for past 6 weeks (total farm, tonnes)
DEMO_ACTUALS = {
    20: {"model": 13.0, "actual": 14.2},
    21: {"model": 24.8, "actual": 24.8},
    22: {"model": 17.1, "actual": 19.1},
    23: {"model": 21.0, "actual": 21.2},
    24: {"model": 21.5, "actual": 23.4},
    25: {"model": 19.8, "actual": 19.6},
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=8, b=0),
    font=dict(family="sans-serif", size=12, color="#5f5e5a"),
    xaxis=dict(showgrid=False, tickcolor="#c3c2b7", linecolor="#c3c2b7"),
    yaxis=dict(gridcolor="#e1e0d9", tickcolor="#c3c2b7", linecolor="rgba(0,0,0,0)"),
    legend=dict(orientation="h", y=1.08, x=0, font_size=11),
    hovermode="x unified",
)


def make_excel(fwd_rows, totals, actuals):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(fwd_rows).to_excel(writer, sheet_name="Forward plan", index=False)

        season = df_fc[["week", "week_date", "block_id", model_col]].copy()
        season = season.rename(columns={model_col: "forecast_t"})
        season.to_excel(writer, sheet_name="Season forecast", index=False)

        log = pd.DataFrame([
            {"week": w, "week_date": week_to_date(w), "model_t": v["model"], "actual_t": v["actual"]}
            for w, v in actuals.items()
        ])
        log.to_excel(writer, sheet_name="Harvest log", index=False)

    return buf.getvalue()


@st.cache_data(show_spinner="Generating training data…")
def load_training_data():
    return build()


@st.cache_resource(show_spinner="Fitting models…")
def train(df_hash):
    df = load_training_data()
    train_df = df[df.season != 2025].reset_index(drop=True)
    theta = fit_baseline(train_df)
    train_df["base"] = base_pred(train_df, theta)
    hybrid = fit_hybrid(train_df, theta)
    return theta, hybrid


df_raw = load_training_data()
theta, hybrid_model = train(len(df_raw))


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sync-badge"><span class="sync-dot"></span> '
        'Climate feed live · updated today 06:14</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.header("Season settings")
    season_plan = st.number_input("Season plan (t)", 100, 2000, 280, step=10)
    picking_cap = st.number_input("Picking capacity (t/week)", 5, 120, 42, step=1)
    horizon = st.slider("Forecast horizon (weeks)", 2, 22, 10)

    st.markdown("---")
    st.header("Model")
    use_hybrid = st.toggle("Hybrid ML correction", value=True,
                           help="Adds a gradient-boosted residual layer on top of the "
                                "mechanistic baseline. Gains ~0.3pp at 4 seasons of data.")

    st.markdown("---")
    st.header("Block configuration")

    DEFAULT_BLOCKS = [
        dict(variety="Florina", plant_week=10, area_m2=8000,  system="glass"),
        dict(variety="Favori",  plant_week=13, area_m2=12000, system="tunnel"),
        dict(variety="Favori",  plant_week=16, area_m2=10000, system="tunnel"),
        dict(variety="Verity",  plant_week=15, area_m2=6000,  system="stellage"),
    ]

    blocks = []
    for i, d in enumerate(DEFAULT_BLOCKS):
        label = f"Block {chr(65+i)}"
        with st.expander(f"**{label}** · {d['system']}", expanded=(i == 0)):
            variety = st.selectbox("Variety", list(VARIETIES), index=list(VARIETIES).index(d["variety"]), key=f"var_{i}")
            system  = st.selectbox("System",  list(SYSTEMS),  index=list(SYSTEMS).index(d["system"]),   key=f"sys_{i}")
            pw      = st.slider("Plant week", 8, 24, d["plant_week"], key=f"pw_{i}")
            area    = st.number_input("Area (m²)", 500, 50000, d["area_m2"], step=500, key=f"area_{i}")

            # GDD to next flush
            from simulator import flush_curve, TBASE
            rng_tmp = np.random.default_rng(pw)
            p = VARIETIES[variety]
            tau_now = sum(max(0, (10.0 + 9.5 * np.sin(2 * np.pi * (w * 7 - 105) / 365)) - TBASE) * 7
                          for w in range(pw, CURRENT_WEEK + 1))
            nxt = next((round(p["tau0"] + k * p["delta"] - tau_now)
                        for k in range(5) if p["tau0"] + k * p["delta"] > tau_now), None)
            if nxt:
                st.caption(f"Next flush ~{nxt} GDD away")

            blocks.append(dict(variety=variety, system=system, plant_week=pw, area_m2=area))


# ── Forecast ─────────────────────────────────────────────────────────────────
df_fc = make_forecast(
    blocks, theta,
    hybrid_model if use_hybrid else None,
    current_week=CURRENT_WEEK,
    horizon=horizon,
)

model_col = "hybrid_t" if use_hybrid else "base_t"

plan_per_week = season_plan / 39  # ~39 productive weeks

if df_fc.empty:
    st.warning("No forecast data — adjust block configuration.")
    st.stop()

totals = df_fc.groupby(["week", "week_date", "is_past"])[[model_col]].sum().reset_index()
totals = totals.rename(columns={model_col: "tonnes"})

fc_season_total = df_fc.groupby("block_id")[model_col].sum().sum()
harvested = sum(v["actual"] for v in DEMO_ACTUALS.values())
remaining = max(0, fc_season_total - harvested)

# Last year (88% scale, shifted planting)
rng_ly = np.random.default_rng(99)
wx_ly  = weather(2025, rng_ly)
ly_rows = []
shifts = [1, -1, 2, 0]
for i, b in enumerate(blocks):
    g = block_frame(2025, b["system"], b["variety"],
                    max(8, b["plant_week"] + shifts[i]), b["area_m2"], wx_ly, rng_ly, noise=False)
    g[model_col] = base_pred(g, theta) * 0.88 / 1000
    ly_rows.append(g)
ly_df = pd.concat(ly_rows, ignore_index=True)
ly_tot = ly_df.groupby("week")[model_col].sum().reset_index()

# Peak week
pk = totals[totals.week > CURRENT_WEEK].sort_values("tonnes", ascending=False)
peak_week = pk.iloc[0]["week"] if len(pk) else CURRENT_WEEK + 1
peak_val  = pk.iloc[0]["tonnes"] if len(pk) else 0.0

# ── KPIs ─────────────────────────────────────────────────────────────────────
st.title("Season forecast 2026")
st.markdown(f"**Fruchthof Hensen** · Week {CURRENT_WEEK} · 4 blocks"
            + (" · **Hybrid ML**" if use_hybrid else " · Mechanistic only"))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Season plan", f"{season_plan} t")
c2.metric("Season forecast", f"{fc_season_total:.0f} t",
          delta=f"{fc_season_total - season_plan:+.0f} t vs plan")
c3.metric("Harvested to date", f"{harvested:.0f} t",
          delta=f"{harvested - (ly_tot[ly_tot.week <= CURRENT_WEEK][model_col].sum()):.0f} t vs last yr")
c4.metric("Still to come", f"{remaining:.0f} t")
c5.metric("Next peak", week_to_date(int(peak_week)), delta=f"{peak_val:.0f} t expected")

st.markdown("---")

# ── Chart 1: Cumulative ───────────────────────────────────────────────────────
all_weeks = sorted(totals.week.unique())
split_week = max(DEMO_ACTUALS.keys())

# Build cumulative using actuals up to split_week, model beyond
cum_vals, cum_acc = [], 0.0
for w in all_weeks:
    row = totals[totals.week == w]
    weekly = DEMO_ACTUALS[w]["actual"] if w in DEMO_ACTUALS else (row.tonnes.values[0] if len(row) else 0)
    cum_acc += weekly
    cum_vals.append(cum_acc)

cum_df = pd.DataFrame({"week": all_weeks, "week_date": [week_to_date(w, return_date=True) for w in all_weeks], "cum": cum_vals})
split_idx = next((i for i, w in enumerate(all_weeks) if w > split_week), len(all_weeks))

# Plan cumulative
plan_cum = [plan_per_week * (i + 1) for i in range(len(all_weeks))]

# Last year cumulative
ly_cum, la = [], 0.0
for w in all_weeks:
    r = ly_tot[ly_tot.week == w][model_col]
    la += r.values[0] if len(r) else 0
    ly_cum.append(la)

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=cum_df.week_date[:split_idx + 1], y=cum_df.cum[:split_idx + 1],
    name="2026 actuals", line=dict(color="#1D9E75", width=2),
    hovertemplate="%{y:.0f} t<extra>actuals</extra>",
))
fig_cum.add_trace(go.Scatter(
    x=cum_df.week_date[split_idx:], y=cum_df.cum[split_idx:],
    name="2026 forecast", line=dict(color="#1D9E75", width=2, dash="dash"),
    hovertemplate="%{y:.0f} t<extra>forecast</extra>",
))
fig_cum.add_trace(go.Scatter(
    x=cum_df.week_date, y=plan_cum,
    name="Season plan", line=dict(color="#378ADD", width=1.5, dash="dot"),
    hovertemplate="%{y:.0f} t<extra>plan</extra>",
))
fig_cum.add_trace(go.Scatter(
    x=cum_df.week_date, y=ly_cum,
    name="2025 actuals", line=dict(color="#B4B2A9", width=1.5, dash="dash"),
    hovertemplate="%{y:.0f} t<extra>2025</extra>",
))
today_date = week_to_date(CURRENT_WEEK, return_date=True)
today_datetime = datetime.combine(today_date, time.min)
fig_cum.add_vline(x=today_datetime, line_width=1, line_dash="dot", line_color="#898781",
                  annotation_text="today", annotation_position="top right",
                  annotation_font_size=10)
fig_cum.update_layout(**PLOTLY_LAYOUT, height=200,
                      yaxis_title="t (cumulative)", xaxis_title=None)
st.subheader("Cumulative harvest — plan vs forecast vs last year")
st.plotly_chart(fig_cum, use_container_width=True)


# ── Chart 2: Actuals card ─────────────────────────────────────────────────────
act_weeks = sorted(DEMO_ACTUALS.keys())
act_labels = [week_to_date(w) for w in act_weeks]
act_model  = [DEMO_ACTUALS[w]["model"]  for w in act_weeks]
act_actual = [DEMO_ACTUALS[w]["actual"] for w in act_weeks]
mape = np.mean([abs(a - m) / a for a, m in zip(act_actual, act_model)]) * 100
bar_colors = ["#1D9E75" if a >= m else "#BA7517" for a, m in zip(act_actual, act_model)]

fig_act = go.Figure()
fig_act.add_trace(go.Bar(
    x=act_labels, y=act_model, name="Model forecast",
    marker_color="rgba(29,158,117,0.20)", marker_line_color="#1D9E75", marker_line_width=1.5,
    hovertemplate="%{y:.1f} t forecast<extra></extra>",
))
fig_act.add_trace(go.Bar(
    x=act_labels, y=act_actual, name="Actual harvested",
    marker_color=bar_colors,
    hovertemplate="%{y:.1f} t actual<extra></extra>",
))
fig_act.update_layout(
    **{**PLOTLY_LAYOUT, "legend": dict(orientation="h", y=1.12, x=0, font_size=11)},
    height=145,
    barmode="overlay",
    xaxis_title=None,
    yaxis_title="t / week",
)

col_act, col_acc = st.columns([3, 1])
with col_act:
    st.subheader("Harvest log — past 6 weeks")
with col_acc:
    st.markdown(f"<br><span style='color:#0ca30c;font-weight:600'>Model within {mape:.0f}% on average</span>",
                unsafe_allow_html=True)
st.plotly_chart(fig_act, use_container_width=True)


# ── Table: 4-week forward ─────────────────────────────────────────────────────
st.subheader("Next 4 weeks — operational planning")
fwd_rows = []
for h in range(1, 5):
    w = CURRENT_WEEK + h
    row = totals[totals.week == w]
    if len(row) == 0:
        continue
    fc = row.tonnes.values[0]
    ly_r = ly_tot[ly_tot.week == w][model_col]
    ly = ly_r.values[0] if len(ly_r) else 0.0
    over = fc > picking_cap
    fwd_rows.append({
        "Week":          week_to_date(w),
        "Forecast (t)":  round(fc, 1),
        "vs plan":       f"{fc - plan_per_week:+.0f} t",
        "vs last year":  f"{fc - ly:+.0f} t",
        "Capacity":      "⚠ over cap" if over else "✓ ok",
    })

if fwd_rows:
    fwd_df = pd.DataFrame(fwd_rows).set_index("Week")
    st.dataframe(
        fwd_df.style.map(
            lambda v: "color: #fab219; font-weight: 600" if "over" in str(v)
            else ("color: #0ca30c" if "✓" in str(v)
            else ("color: #0ca30c" if str(v).startswith("+") else
                  ("color: #e24b4a" if str(v).startswith("-") else ""))),
        ),
        use_container_width=True,
    )
    st.download_button(
        label="Download forecast (.xlsx)",
        data=make_excel(fwd_rows, df_fc, DEMO_ACTUALS),
        file_name=f"hexafarms_forecast_W{CURRENT_WEEK}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("---")

# ── Chart 3: Weekly by block ──────────────────────────────────────────────────
block_ids = sorted(df_fc.block_id.unique())
all_wk_dates = sorted(df_fc.week_date.unique(), key=lambda s: df_fc[df_fc.week_date == s].week.values[0])

fig_wk = go.Figure()
for bi, bid in enumerate(block_ids):
    sub = df_fc[df_fc.block_id == bid].sort_values("week")
    col = BLOCK_COLORS[bi % len(BLOCK_COLORS)]

    past = sub[sub.is_past]
    future = sub[~sub.is_past]

    if len(past):
        fig_wk.add_trace(go.Bar(
            x=past.week_date, y=past[model_col],
            name=bid, marker_color=col, opacity=0.92,
            legendgroup=bid, showlegend=True,
        ))
    if len(future):
        fig_wk.add_trace(go.Bar(
            x=future.week_date, y=future[model_col],
            name=bid, marker_color=col, opacity=0.30,
            legendgroup=bid, showlegend=False,
        ))

fig_wk.add_trace(go.Scatter(
    x=all_wk_dates, y=[picking_cap] * len(all_wk_dates),
    name="Picking capacity", mode="lines",
    line=dict(color="#BA7517", width=1.5, dash="dot"),
))
fig_wk.add_vline(x=week_to_date(CURRENT_WEEK), line_width=1, line_dash="dot",
                 line_color="#898781")
fig_wk.update_layout(
    **PLOTLY_LAYOUT, height=230, barmode="stack",
    yaxis_title="t / week", xaxis_title=None,
)
st.subheader("Weekly yield by block")
st.plotly_chart(fig_wk, use_container_width=True)

st.caption(
    "Model: mechanistic flush curve in thermal time (Tbase 3°C)" +
    (" + gradient-boosted residual (HistGBM, 4 seasons)" if use_hybrid else "") +
    " · Synthetic training data · Perfect weather assumption (upper bound)"
)

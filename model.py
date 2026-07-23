"""
Two-stage yield model:
  Stage 1 — mechanistic baseline: flush curve in thermal time, fitted by least-squares.
  Stage 2 — learned residual: HGB or XGBoost on (actual − baseline).

At ~4 seasons of data the two models are roughly equivalent (≈7% MAPE on farm-level
weekly total). The ML layer earns its keep as data accumulates and picks up block
quality effects the physics cannot see.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor

from simulator import SYSTEMS, VARIETIES, TBASE, N_FLUSH, ALPHA, flush_curve, flush_rate, weather, block_frame

FEAT_COLS = [
    "week", "weeks_since_plant", "tau", "gdd", "dli",
    "tmax", "tmean", "plants", "area_m2", "plant_week",
]


def _density(tau, tau0, delta, sigma, rho):
    y = np.zeros_like(tau, dtype=float)
    for k in range(N_FLUSH):
        y += (rho ** k) * np.exp(-((tau - (tau0 + k * delta)) ** 2) / (2 * sigma ** 2))
    return y


def base_pred(df, theta):
    varieties = sorted(df.variety.unique())
    alpha = theta[-1]
    out = np.zeros(len(df))
    for vi, v in enumerate(varieties):
        tau0, delta, sigma, rho, r = theta[vi * 5:(vi + 1) * 5]
        m = (df.variety == v).values
        out[m] = (
            df.loc[m, "plants"].values * r
            * _density(df.loc[m, "tau"].values, tau0, delta, sigma, rho)
            * df.loc[m, "gdd"].values
            * (df.loc[m, "dli"].values / 14.0) ** alpha
        )
    return out


def fit_baseline(train):
    varieties = sorted(train.variety.unique())
    nv = len(varieties)
    x0 = np.concatenate([np.tile([500., 400., 100., 0.85, 9e-4], nv), [0.5]])
    lo = np.concatenate([np.tile([250., 200., 40., 0.4, 1e-5], nv), [0.0]])
    hi = np.concatenate([np.tile([900., 700., 250., 1.2, 1e-2], nv), [2.0]])

    def resid(th):
        p = base_pred(train, th)
        return np.sqrt(np.maximum(p, 0) + 1) - np.sqrt(train.kg.values + 1)

    res = least_squares(resid, x0, bounds=(lo, hi), max_nfev=1500)
    return res.x


def get_features(df):
    x = df[FEAT_COLS].copy()
    for s in sorted(SYSTEMS):
        x[f"sys_{s}"] = (df.system == s).astype(int)
    for v in sorted(VARIETIES):
        x[f"var_{v}"] = (df.variety == v).astype(int)
    return x


def fit_hybrid(train, theta):
    train = train.copy()
    train["base"] = base_pred(train, theta)
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6, random_state=0
    )
    model.fit(get_features(train), train.kg - train.base)
    return model


def week_to_date(week, year=2026, return_date=False):
    from datetime import date, timedelta
    week = int(week)
    d = date(year, 1, 1) + timedelta(weeks=week - 1)
    if return_date:
        return d
    return d.strftime("%-d %b")


def make_forecast(blocks, theta, hybrid_model, current_week=26, horizon=10, seed=42):
    """
    Generate a per-week, per-block forecast DataFrame.

    blocks: list of dicts with keys variety, plant_week, area_m2, system
    Returns a DataFrame ready for Plotly charts.
    """
    rng = np.random.default_rng(seed)
    wx = weather(2026, rng)

    rows = []
    for bi, b in enumerate(blocks):
        g = block_frame(
            2026, b["system"], b["variety"],
            b["plant_week"], b["area_m2"], wx, rng, noise=False,
        )
        if len(g) == 0:
            continue

        g["base_kg"] = base_pred(g, theta)

        if hybrid_model is not None:
            correction = hybrid_model.predict(get_features(g))
            g["hybrid_kg"] = np.clip(g.base_kg + correction, 0, None)
        else:
            g["hybrid_kg"] = g.base_kg

        g["block_id"] = f"Block {chr(65 + bi)}"
        g["block_label"] = f"Block {chr(65 + bi)} ({b['system']})"
        rows.append(g)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    max_week = current_week + horizon
    df = df[df.week <= max_week].copy()
    df["week_date"] = df.week.apply(week_to_date)
    df["is_past"] = df.week <= current_week
    df["base_t"] = df.base_kg / 1000
    df["hybrid_t"] = df.hybrid_kg / 1000
    return df

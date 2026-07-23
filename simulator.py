"""
Synthetic strawberry harvest data for the hexafarms demo.

Ground truth: mechanistic flush curve in thermal time (GDD, Tbase=3°C).
Temperature sets *when* the flush peaks; light sets *how much*.
"""
import numpy as np
import pandas as pd

TBASE = 3.0
SEASON_END_DOY = 345
N_FLUSH = 5
ALPHA = 0.55
SEASONS = [2021, 2022, 2023, 2024, 2025]

SYSTEMS = {
    "glass":    dict(dT=6.0, trans=0.72, dens=9.5),
    "tunnel":   dict(dT=3.0, trans=0.82, dens=8.5),
    "stellage": dict(dT=1.2, trans=0.90, dens=8.0),
    "field":    dict(dT=0.0, trans=1.00, dens=6.5),
}

VARIETIES = {
    "Florina": dict(tau0=520, delta=400, sigma=95,  rho=0.84, kgp=0.92),
    "Favori":  dict(tau0=470, delta=370, sigma=105, rho=0.88, kgp=0.85),
    "Verity":  dict(tau0=560, delta=430, sigma=88,  rho=0.80, kgp=0.78),
}


def flush_curve(tau, p):
    y = np.zeros_like(tau, dtype=float)
    for k in range(N_FLUSH):
        y += (p["rho"] ** k) * np.exp(
            -((tau - (p["tau0"] + k * p["delta"])) ** 2) / (2 * p["sigma"] ** 2)
        )
    return y


def flush_rate(p):
    integral = sum(p["rho"] ** k for k in range(N_FLUSH)) * p["sigma"] * np.sqrt(2 * np.pi)
    return p["kgp"] / integral


def weather(season, rng):
    doy = np.arange(1, 366)
    base = 10.0 + 9.5 * np.sin(2 * np.pi * (doy - 105) / 365)
    w = np.zeros(365)
    for i in range(1, 365):
        w[i] = 0.78 * w[i - 1] + rng.normal(0, 1.9)
    tmean = base + rng.normal(0, 1.1) + w
    amp = 6.5 + 2.0 * np.sin(2 * np.pi * (doy - 105) / 365)
    dli = np.clip(
        22 * np.sin(np.pi * np.clip((doy - 60) / 250, 0, 1)) + rng.normal(0, 2.2, 365), 1, None
    )
    return pd.DataFrame(dict(
        season=season, doy=doy,
        tmin=tmean - amp / 2, tmax=tmean + amp / 2, dli_out=dli,
    ))


def block_frame(season, system, variety, plant_week, area, wx, rng, noise=True):
    S, P = SYSTEMS[system], VARIETIES[variety]
    plants = area * S["dens"]
    d = wx.copy()
    d["tmean_in"] = (d.tmin + d.tmax) / 2 + S["dT"]
    d["gdd"] = np.clip(d.tmean_in - TBASE, 0, None)
    d["dli_in"] = d.dli_out * S["trans"]
    d = d[(d.doy >= plant_week * 7) & (d.doy <= SEASON_END_DOY)].copy()
    d["tau"] = d.gdd.cumsum()

    kg = (
        plants * flush_rate(P)
        * flush_curve(d.tau.values, P)
        * d.gdd.values
        * (d.dli_in.values / 14.0) ** ALPHA
    )
    kg *= np.clip(1 - 0.045 * np.clip(d.tmax.values + S["dT"] - 28, 0, None), 0.5, 1.0)
    if noise:
        kg *= np.exp(rng.normal(0, 0.11))

    d["kg"] = kg
    d["week"] = ((d.doy - 1) // 7) + 1
    g = (
        d.groupby("week")
        .agg(kg=("kg", "sum"), gdd=("gdd", "sum"), tau=("tau", "last"),
             dli=("dli_in", "mean"), tmax=("tmax", "max"), tmean=("tmean_in", "mean"))
        .reset_index()
    )
    g = g[g.kg > 5].reset_index(drop=True)

    if noise and len(g) > 1:
        e = np.zeros(len(g))
        for i in range(1, len(g)):
            e[i] = 0.45 * e[i - 1] + rng.normal(0, 0.115)
        shock = np.where(rng.random(len(g)) < 0.035, rng.uniform(0.55, 0.8, len(g)), 1.0)
        g["kg"] = g.kg * np.exp(e) * shock

    g["season"] = season
    g["block"] = f"{season}-{system}-{plant_week}-{int(area)}"
    g["system"] = system
    g["variety"] = variety
    g["plant_week"] = plant_week
    g["area_m2"] = area
    g["plants"] = plants
    g["weeks_since_plant"] = g.week - plant_week
    return g


def build(seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for season in SEASONS:
        wx = weather(season, rng)
        for b in range(14):
            rows.append(block_frame(
                season,
                list(SYSTEMS)[b % 4],
                list(VARIETIES)[b % 3],
                int(rng.integers(10, 23)),
                float(rng.integers(5000, 20000)),
                wx, rng,
            ))
    return pd.concat(rows, ignore_index=True)

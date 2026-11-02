"""
El Nino seasonal prediction model for Nairobi, Kenya.

Predicts monthly precipitation and temperature anomalies from lagged
climate indices (ONI, IOD/DMI, SOI) using ridge regression and random
forest, with rolling-origin cross-validation, then produces an
Oct 2026 - Mar 2027 forecast under a persisted strong-El Nino scenario.

Data (fetched with fetch_data.sh):
  data/oni.txt            NOAA CPC Oceanic Nino Index (to Aug 2026)
  data/dmi.txt            NOAA PSL Dipole Mode Index, HadISST (to Jul 2026)
  data/soi.txt            NOAA CPC Southern Oscillation Index (to Dec 2025)
  data/power_nairobi.json NASA POWER monthly precip/temp, 1.28S 36.82E, 1981-2025
  data/nairobi_daily.csv  GHCN daily station KE000063740 (Jomo Kenyatta Intl)

Precipitation: NASA POWER (GHCN station gauge coverage collapses after
~2000, validated against the station record over 1985-1998 overlap).
Temperature: GHCN Nairobi station (TAVG through 2025).
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA = "data"
CLIM_START, CLIM_END = 1991, 2020
FORECAST_MONTHS = pd.period_range("2026-10", "2027-03", freq="M")
STATION_FILE = f"{DATA}/nairobi_daily.csv"


# ---------- data parsing ----------

def load_oni(path):
    season_center = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5,
                     "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10,
                     "OND": 11, "NDJ": 12}
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 4 and p[0] in season_center:
                rows.append((int(p[1]), season_center[p[0]], float(p[3])))
    df = pd.DataFrame(rows, columns=["year", "month", "oni"])
    return df.set_index(pd.PeriodIndex(
        pd.to_datetime(dict(year=df.year, month=df.month, day=1)), freq="M"))["oni"]


def load_soi(path):
    rows, in_data = [], False
    with open(path) as f:
        for line in f:
            p = line.split()
            if p and p[0] == "YEAR":
                if in_data:
                    break  # first table (SOI) only; file stacks Tahiti/Darwin next
                in_data = True
                continue
            if in_data and len(p) == 13:
                try:
                    vals = [float(x) for x in p[1:]]
                except ValueError:
                    continue
                rows.append((int(p[0]), vals))
    recs = [(y, m + 1, v) for y, vals in rows for m, v in enumerate(vals)]
    df = pd.DataFrame(recs, columns=["year", "month", "soi"])
    return df.set_index(pd.PeriodIndex(
        pd.to_datetime(dict(year=df.year, month=df.month, day=1)), freq="M"))["soi"]


def load_dmi(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 13:
                try:
                    vals = [float(x) for x in p[1:]]
                except ValueError:
                    continue
                rows.append((int(p[0]), vals))
    recs = [(y, m + 1, v) for y, vals in rows for m, v in enumerate(vals) if v > -9000]
    df = pd.DataFrame(recs, columns=["year", "month", "dmi"])
    return df.set_index(pd.PeriodIndex(
        pd.to_datetime(dict(year=df.year, month=df.month, day=1)), freq="M"))["dmi"]


def load_power(path):
    """NASA POWER monthly point data -> monthly precip mm and mean temp C."""
    raw = json.load(open(path))["properties"]["parameter"]
    recs = []
    for ym, mm_day in raw["PRECTOTCORR"].items():
        if mm_day < -900 or len(ym) != 6 or int(ym[4:]) > 12:  # skip -9999 and annual aggregates
            continue
        per = pd.Period(f"{ym[:4]}-{ym[4:]}", freq="M")
        days = per.days_in_month
        recs.append((per, mm_day * days, raw["T2M"][ym]))
    df = pd.DataFrame(recs, columns=["month", "prcp_mm", "t2m_c"]).set_index("month")
    return df


def load_ghcn_monthly(path, elements=("PRCP", "TAVG")):
    cols = ["station", "date", "element", "value", "mflag", "qflag", "sflag", "obs"]
    daily = pd.read_csv(path, names=cols, dtype=str, low_memory=False)
    daily = daily[(daily.element.isin(elements)) & (daily.qflag.isna())
                  & (daily.value.astype(float) != -9999)]
    daily["date"] = pd.to_datetime(daily.date, format="%Y%m%d")
    daily["month"] = daily.date.dt.to_period("M")
    daily["value"] = daily.value.astype(float) / 10.0
    out = {}
    for el in elements:
        g = daily[daily.element == el].groupby("month")["value"]
        s = g.sum() if el == "PRCP" else g.mean()
        days = g.size()
        out[el.lower()] = s.where(days >= 25)
    return pd.DataFrame(out)


# ---------- dataset ----------

def build_dataset():
    power = load_power(f"{DATA}/power_nairobi.json")
    ghcn = load_ghcn_monthly(STATION_FILE)
    oni, dmi, soi = (load_oni(f"{DATA}/oni.txt"),
                     load_dmi(f"{DATA}/dmi.txt"),
                     load_soi(f"{DATA}/soi.txt"))
    idx = pd.DataFrame({"oni": oni, "dmi": dmi, "soi": soi})
    idx = idx[~idx.index.duplicated(keep="last")].sort_index()

    # overlap validation: POWER satellite-gauge vs station gauge
    ov = power.join(ghcn[["prcp"]].rename(columns={"prcp": "ghcn_prcp"}), how="inner").dropna()
    ov = ov[(ov.index.year >= 1985) & (ov.index.year <= 1998)]
    r_val = np.corrcoef(ov.prcp_mm, ov.ghcn_prcp)[0, 1]

    full = pd.period_range("1981-01", FORECAST_MONTHS[-1], freq="M")
    df = pd.DataFrame(index=full)
    df = df.join(idx)
    df = df.join(power[["prcp_mm", "t2m_c"]])
    df = df.join(ghcn[["tavg"]].rename(columns={"tavg": "tavg_station"}))
    # station temp preferred, POWER 2m temp fills gaps
    df["tavg_c"] = df.tavg_station.fillna(df.t2m_c)

    # forecast-period indices: persist last observed state (El Nino scenario)
    last_idx = {c: df[c].dropna().index[-1] for c in ["oni", "dmi", "soi"]}
    for c in ["oni", "dmi", "soi"]:
        df.loc[df.index > last_idx[c], c] = df[c].dropna().iloc[-1]

    for k in range(1, 7):
        df[f"oni_lag{k}"] = df.oni.shift(k)
    for k in range(1, 4):
        df[f"dmi_lag{k}"] = df.dmi.shift(k)
        df[f"soi_lag{k}"] = df.soi.shift(k)
    df["sin_m"] = np.sin(2 * np.pi * df.index.month / 12)
    df["cos_m"] = np.cos(2 * np.pi * df.index.month / 12)
    # El Nino/IOD effects are season-dependent in East Africa (OND short rains)
    df["oni_x_sin"] = df.oni_lag1 * df.sin_m
    df["oni_x_cos"] = df.oni_lag1 * df.cos_m
    df["dmi_x_sin"] = df.dmi_lag1 * df.sin_m
    df["dmi_x_cos"] = df.dmi_lag1 * df.cos_m

    clim = df[(df.index.year >= CLIM_START) & (df.index.year <= CLIM_END)]
    prcp_clim = clim.groupby(clim.index.month).prcp_mm.mean()
    temp_clim = clim.groupby(clim.index.month).tavg_c.mean()
    df["prcp_anom"] = df.prcp_mm - df.index.month.map(prcp_clim)
    df["temp_anom"] = df.tavg_c - df.index.month.map(temp_clim)

    ter = clim.groupby(clim.index.month).prcp_mm.quantile([1 / 3, 2 / 3]).unstack()
    df["prcp_tercile"] = np.select(
        [df.prcp_mm < df.index.month.map(ter[1 / 3]),
         df.prcp_mm > df.index.month.map(ter[2 / 3])],
        ["below", "above"], default="normal")
    return df, {"prcp_clim": prcp_clim, "temp_clim": temp_clim,
                "terciles": ter, "r_power_ghcn": r_val,
                "last_idx": last_idx,
                "state": {c: df[c].dropna().iloc[-1] for c in ["oni", "dmi", "soi"]}}


FEATURES = (["oni_lag1", "oni_lag2", "oni_lag3", "oni_lag4", "oni_lag5", "oni_lag6"]
            + ["dmi_lag1", "dmi_lag2", "dmi_lag3"]
            + ["soi_lag1", "soi_lag2", "soi_lag3"]
            + ["sin_m", "cos_m"]
            + ["oni_x_sin", "oni_x_cos", "dmi_x_sin", "dmi_x_cos"])
TARGETS = ["prcp_anom", "temp_anom"]


# Seasons that matter for the Oct 2026 - Mar 2027 outlook window.
# label -> (months, anchor month whose pre-season indices predict it)
SEASONS = {"OND": ([10, 11, 12], 9), "DJF": ([12, 1, 2], 11), "JFM": ([1, 2, 3], 12)}


def build_seasons(df):
    """Per-year seasonal precip anomaly sums with pre-season index state."""
    hist = df[df.index <= pd.Period("2025-12")]
    rows = []
    for y in range(1981, 2026):
        for label, (months, anchor_m) in SEASONS.items():
            # OND: y,y,y ; DJF: y-1(Dec),y,y ; JFM: y-1(Jan,Feb),y(Mar)
            if label == "OND":
                per = [pd.Period(f"{y}-{m:02d}") for m in months]
            elif label == "DJF":
                per = [pd.Period(f"{y-1}-12"), pd.Period(f"{y}-01"),
                       pd.Period(f"{y}-02")]
            else:
                per = [pd.Period(f"{y-1}-01"), pd.Period(f"{y-1}-02"),
                       pd.Period(f"{y}-03")]
            vals = hist.prcp_anom.reindex(per)
            if vals.isna().any():
                continue
            anchor = per[0] - 1
            rows.append({"season": label, "year": y, "prcp_anom": vals.sum(),
                         "oni": df.oni.get(anchor, np.nan),
                         "dmi": df.dmi.get(anchor, np.nan),
                         "soi": df.soi.get(anchor, np.nan)})
    return pd.DataFrame(rows)


IDX_FEATS = ["oni", "dmi", "soi"]


def temp_cv(df):
    """Monthly temperature anomaly, rolling-origin CV."""
    folds = [(2000, 2005), (2006, 2011), (2012, 2017), (2018, 2025)]
    rows = []
    for tr, te in folds:
        train = df[df.index.year < tr].dropna(subset=FEATURES + ["temp_anom"])
        test = df[(df.index.year >= tr) & (df.index.year < te)].dropna(
            subset=FEATURES + ["temp_anom"])
        for name, make in [("ridge", lambda: Pipeline([
                ("sc", StandardScaler()),
                ("m", RidgeCV(alphas=np.logspace(-2, 3, 50)))])),
                ("rf", lambda: RandomForestRegressor(
                    n_estimators=400, min_samples_leaf=3, random_state=7,
                    n_jobs=-1))]:
            mdl = make().fit(train[FEATURES], train.temp_anom)
            p = mdl.predict(test[FEATURES])
            rows.append({"fold": f"{tr}-{te-1}", "model": name,
                         "rmse": round(np.sqrt(np.mean((p - test.temp_anom) ** 2)), 2),
                         "corr": round(np.corrcoef(p, test.temp_anom)[0, 1], 2),
                         "n": len(test)})
    return pd.DataFrame(rows)


def seasonal_cv(seas):
    """Per-season expanding-window CV of the seasonal precip model."""
    folds = [(2000, 2005), (2006, 2011), (2012, 2017), (2018, 2025)]
    rows, preds = [], []
    for tr, te in folds:
        for label in SEASONS:
            d = seas[seas.season == label].dropna(subset=IDX_FEATS + ["prcp_anom"])
            train, test = d[d.year < tr], d[(d.year >= tr) & (d.year < te)]
            if len(train) < 10 or len(test) < 3:
                continue
            mdl = Pipeline([("sc", StandardScaler()),
                            ("m", RidgeCV(alphas=np.logspace(-2, 3, 50)))])
            mdl.fit(train[IDX_FEATS], train.prcp_anom)
            p = mdl.predict(test[IDX_FEATS])
            rows.append({"season": label, "fold": f"{tr}-{te-1}",
                         "r": round(np.corrcoef(p, test.prcp_anom)[0, 1], 2),
                         "n": len(test)})
            preds.append(pd.DataFrame({"season": label, "year": test.year.values,
                                       "obs": test.prcp_anom.values, "pred": p}))
    return pd.DataFrame(rows), pd.concat(preds)


def season_clim_terciles(seas):
    clim = seas[seas.year.between(CLIM_START, CLIM_END)]
    return clim.groupby("season").prcp_anom.quantile([1 / 3, 2 / 3]).unstack()


def main():
    os.makedirs("output", exist_ok=True)
    df, meta = build_dataset()
    hist = df[df.index <= pd.Period("2025-12")]
    hist.to_csv(f"{DATA}/monthly_dataset.csv")

    print("=== Dataset ===")
    v = hist.dropna(subset=["prcp_mm", "tavg_c"])
    print(f"Monthly series {v.index.min()} to {v.index.max()}, {len(v)} months "
          f"(precip: NASA POWER; temp: GHCN Nairobi, POWER-filled)")
    print(f"Validation, POWER vs GHCN gauge precip 1985-1998: r = {meta['r_power_ghcn']:.2f}")
    print(f"Index state (persisted for forecast): "
          f"ONI({meta['last_idx']['oni']}) = {meta['state']['oni']:+.2f} | "
          f"DMI({meta['last_idx']['dmi']}) = {meta['state']['dmi']:+.2f} | "
          f"SOI({meta['last_idx']['soi']}) = {meta['state']['soi']:+.2f}")

    # --- temperature: monthly model ---
    tcv = temp_cv(hist)
    print("\n=== Temperature model CV (monthly anomalies, 4 folds) ===")
    print(tcv.pivot_table(index="model", values=["rmse", "corr"],
                          aggfunc="mean").round(2).to_string())
    train_t = hist[(hist.index.year >= 1985) & (hist.index.year <= 2024)].dropna(
        subset=FEATURES + ["temp_anom"])
    t_ridge = Pipeline([("sc", StandardScaler()),
                        ("m", RidgeCV(alphas=np.logspace(-2, 3, 50)))
                        ]).fit(train_t[FEATURES], train_t.temp_anom)
    t_rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                 random_state=7, n_jobs=-1
                                 ).fit(train_t[FEATURES], train_t.temp_anom)

    fc_df = df.loc[FORECAST_MONTHS].dropna(subset=FEATURES)
    fc = pd.DataFrame(index=fc_df.index)
    fc["temp_anom_ridge"] = t_ridge.predict(fc_df[FEATURES])
    fc["temp_anom_rf"] = t_rf.predict(fc_df[FEATURES])
    fc["temp_anom"] = fc[["temp_anom_ridge", "temp_anom_rf"]].mean(axis=1)

    # --- precipitation: seasonal model ---
    seas = build_seasons(df)
    scv, spred = seasonal_cv(seas)
    print("\n=== Precipitation model CV (seasonal anomalies, ridge) ===")
    print(scv.to_string(index=False))

    ter_s = season_clim_terciles(seas)
    state = meta["state"]
    fc_rows = []
    # forecast season spans and their pre-season anchor months
    fc_spans = {"OND": [pd.Period("2026-10"), pd.Period("2026-11"), pd.Period("2026-12")],
                "DJF": [pd.Period("2026-12"), pd.Period("2027-01"), pd.Period("2027-02")],
                "JFM": [pd.Period("2027-01"), pd.Period("2027-02"), pd.Period("2027-03")]}
    for label, span in fc_spans.items():
        d = seas[seas.season == label].dropna(subset=IDX_FEATS + ["prcp_anom"])
        mdl = Pipeline([("sc", StandardScaler()),
                        ("m", RidgeCV(alphas=np.logspace(-2, 3, 50)))])
        mdl.fit(d[IDX_FEATS], d.prcp_anom)
        anchor = span[0] - 1
        x = pd.DataFrame([{c: df[c].get(anchor, state[c]) for c in IDX_FEATS}])
        pred = mdl.predict(x)[0]
        # ONI-only linear benchmark
        o = np.polyfit(d.oni, d.prcp_anom, 1)
        bench = np.polyval(o, x.oni)[0]
        lo, hi = ter_s.loc[label, 1 / 3], ter_s.loc[label, 2 / 3]
        cls = ("above normal" if pred > hi else
               "below normal" if pred < lo else "near normal")
        fc_rows.append({"season": label, "oni": round(x.oni[0], 2),
                        "prcp_anom": round(pred, 1), "oni_only_bench": round(bench, 1),
                        "t1/3": round(lo, 1), "t2/3": round(hi, 1),
                        "class": cls})
    sfc = pd.DataFrame(fc_rows).set_index("season")

    # allocate seasonal precip anomaly to months by climatological share
    clim_m = meta["prcp_clim"]
    fc["prcp_anom"] = np.nan
    for label, months in (("OND", [10, 11, 12]), ("JFM", [1, 2, 3])):
        share = {m: clim_m[m] / clim_m[months].sum() for m in months}
        anom = sfc.loc[label, "prcp_anom"]
        for m in months:
            fc.loc[fc.index.month == m, "prcp_anom"] = anom * share[m]
    fc["prcp_total"] = fc.index.month.map(clim_m) + fc.prcp_anom

    print("\n=== Forecast: Oct 2026 - Mar 2027 (persisted strong El Nino scenario) ===")
    print("\nSeasonal precipitation (ridge on ONI+DMI+SOI):")
    print(sfc.to_string())
    print("\nMonthly view (seasonal anomaly allocated by climatological share):")
    print(fc[["prcp_anom", "prcp_total", "temp_anom_ridge", "temp_anom_rf",
              "temp_anom"]].round(1).to_string())
    print(f"\nTemperature: seasonal means (OND {fc.temp_anom[fc.index.month.isin([10,11,12])].mean():+.2f}°C, "
          f"JFM {fc.temp_anom[fc.index.month.isin([1,2,3])].mean():+.2f}°C)")

    fc.to_csv("output/forecast_monthly.csv")
    sfc.to_csv("output/forecast_seasonal.csv")
    tcv.to_csv("output/cv_temperature.csv", index=False)
    scv.to_csv("output/cv_precip_seasonal.csv", index=False)
    spred.to_csv("output/cv_precip_predictions.csv", index=False)

    # --- charts ---
    x = np.arange(len(fc))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].bar(x, fc.prcp_anom, 0.5, color="steelblue")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("Precip anomaly (mm)")
    axes[0].set_title("Nairobi precipitation anomaly — OND 2026: "
                      f"{sfc.loc['OND','prcp_anom']:+.0f} mm ({sfc.loc['OND','class']}), "
                      f"JFM 2027: {sfc.loc['JFM','prcp_anom']:+.0f} mm ({sfc.loc['JFM','class']})")
    axes[1].bar(x - 0.18, fc.temp_anom_ridge, 0.36, label="Ridge")
    axes[1].bar(x + 0.18, fc.temp_anom_rf, 0.36, label="Random forest")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("Temp anomaly (°C)")
    axes[1].set_title("Nairobi monthly mean temperature anomaly "
                      f"(ONI {meta['state']['oni']:+.1f} persisted)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fc.index.astype(str), rotation=30)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("output/forecast.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, c in zip(SEASONS, ["tab:blue", "tab:orange", "tab:green"]):
        s = spred[spred.season == label]
        ax.scatter(s.obs, s.pred, s=20, alpha=0.7, color=c, label=label)
    lim = max(abs(spred.obs).max(), abs(spred.pred).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set_xlabel("Observed seasonal precip anomaly (mm)")
    ax.set_ylabel("Predicted (out-of-sample)")
    ax.set_title("Seasonal precip model, cross-validated")
    ax.legend()
    plt.tight_layout()
    plt.savefig("output/cv_skill.png", dpi=150)
    print("\nCharts: output/forecast.png, output/cv_skill.png")


if __name__ == "__main__":
    main()

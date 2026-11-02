#!/bin/bash
# Fetch all inputs for kenya_elnino_model.py. Re-run monthly to refresh indices.
set -e
cd "$(dirname "$0")"
mkdir -p data

curl -sL -o data/oni.txt "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
curl -sL -o data/soi.txt "https://www.cpc.ncep.noaa.gov/data/indices/soi"
curl -sL -o data/dmi.txt "https://psl.noaa.gov/data/timeseries/month/data/dmi.had.long.data"
curl -sL -o data/nairobi_daily.csv.gz "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/KE000063740.csv.gz"
gunzip -f data/nairobi_daily.csv.gz

# NASA POWER monthly precip + 2m temp at the outlook point 1.28S, 36.82E
curl -sL -o data/power_nairobi.json \
  "https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOTCORR,T2M&community=AG&longitude=36.82&latitude=-1.28&start=1981&end=2025&format=JSON"

python3 kenya_elnino_model.py

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

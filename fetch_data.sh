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

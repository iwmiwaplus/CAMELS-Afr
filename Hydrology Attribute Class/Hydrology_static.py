import pandas as pd
import numpy as np
from scipy.stats import skew
import re
 
# ======================================================
# FILE PATHS
# ======================================================
q_fp    = r"...\CAMELS_Afr_Hydrology_L1_q.csv"
p_fp    = r"...\CAMELS_Afr_Climate_Monthly_L1_p_chirps.csv"
area_fp = r"...\CAMELS_Afr_Location_Static_L1.csv"
 
out_fp  = r"...\CAMELS_Afr_Hydrology_Static_L1.csv"
 
# ======================================================
# LOAD DATA (LONG FORMAT)
# ======================================================
q_long = pd.read_csv(q_fp, parse_dates=["Month"])
p_long = pd.read_csv(p_fp, parse_dates=["Month"])
area_df = pd.read_csv(area_fp)
 
# ======================================================
# SET INDEX
# ======================================================
q_long = q_long.set_index("Month")
p_long = p_long.set_index("Month")
 
# ======================================================
# ALIGN TIME
# ======================================================
common_time = q_long.index.intersection(p_long.index)
q_long = q_long.loc[common_time]
p_long = p_long.loc[common_time]
 
# ======================================================
# BASINS
# ======================================================
basins = q_long.columns.tolist()
 
# ======================================================
# CONVERT TO MATRIX
# ======================================================
q_matrix = q_long[basins].apply(pd.to_numeric, errors="coerce").values
p_matrix = p_long[basins].apply(pd.to_numeric, errors="coerce").values
 
n_time = q_matrix.shape[0]
 
# ======================================================
# CORE METRICS
# ======================================================
q_mean = np.nanmean(q_matrix, axis=0)
annual_q = np.nansum(q_matrix, axis=0) / (n_time / 12)
 
# Percentiles
q_5  = np.nanpercentile(q_matrix, 5, axis=0)
q_10 = np.nanpercentile(q_matrix, 10, axis=0)
q_25 = np.nanpercentile(q_matrix, 25, axis=0)
q_50 = np.nanpercentile(q_matrix, 50, axis=0)
q_75 = np.nanpercentile(q_matrix, 75, axis=0)
q_90 = np.nanpercentile(q_matrix, 90, axis=0)
q_95 = np.nanpercentile(q_matrix, 95, axis=0)
 
# FDC slope
q_33 = np.nanpercentile(q_matrix, 33, axis=0)
q_66 = np.nanpercentile(q_matrix, 66, axis=0)
slope_fdc = (np.log(q_66 + 1e-6) - np.log(q_33 + 1e-6)) / 0.33
 
# Variability
q_cv   = np.nanstd(q_matrix, axis=0) / (q_mean + 1e-6)
q_skew = skew(q_matrix, axis=0, nan_policy='omit')
 
# ======================================================
# HIGH / LOW FLOW
# ======================================================
median_q = np.nanmedian(q_matrix, axis=0)
mean_q_long = np.nanmean(q_matrix, axis=0)
 
def get_freq_dur(matrix, threshold, condition='above'):
    freqs, durs = [], []
    for i in range(matrix.shape[1]):
        series = matrix[:, i]
 
        if condition == 'above':
            mask = series > threshold[i]
        else:
            mask = series <= threshold[i]
 
        mask = np.nan_to_num(mask, nan=False)
 
        events = np.diff(mask.astype(int)) == 1
        freqs.append(np.sum(events) / (len(series)/12))
 
        if np.any(mask):
            runs = np.diff(np.where(np.concatenate(([mask[0]], mask[:-1] != mask[1:], [True])))[0])[::2]
            durs.append(np.mean(runs))
        else:
            durs.append(np.nan)
 
    return freqs, durs
 
high_q_freq, high_q_dur = get_freq_dur(q_matrix, 9 * median_q, 'above')
low_q_freq,  low_q_dur  = get_freq_dur(q_matrix, 0.2 * mean_q_long, 'below')
 
zero_q_freq = np.sum(q_matrix == 0, axis=0) / n_time
 
# ======================================================
# RLD
# ======================================================
diffs = np.diff(q_matrix, axis=0)
rld = np.sum(diffs > 0, axis=0) / n_time
 
# ======================================================
# MONTHLY CLIMATOLOGY
# ======================================================
monthly_means = {}
 
for m in range(1, 13):
    mask = q_long.index.month == m
    monthly_means[m] = np.nanmean(q_matrix[mask, :], axis=0)
 
# ======================================================
# CLIMATE METRICS
# ======================================================
mean_p = np.nanmean(p_matrix, axis=0)
 
# ======================================================
# RUNOFF RATIO (AREA IN km²)
# ======================================================
 
# Convert area to m²
area_series = area_df.set_index("Basin_ID").loc[basins]["catchment_area"].values * 1e6
 
# Seconds in a month
seconds_per_month = 60 * 60 * 24 * 30
 
# Convert discharge (m³/s) → runoff depth (mm/month)
runoff_mm = (q_mean * seconds_per_month) / area_series * 1000
 
# Avoid division issues
runoff_mm = np.where(area_series > 0, runoff_mm, np.nan)
 
# Runoff ratio (dimensionless)
runoff_ratio = runoff_mm / mean_p
##########
 
# Elasticity
stream_elas = []
 
for i in range(len(basins)):
    q = q_matrix[:, i]
    p = p_matrix[:, i]
 
    qy = pd.Series(q, index=q_long.index).resample("YE").sum()
    py = pd.Series(p, index=p_long.index).resample("YE").sum()
 
    df = pd.concat([qy, py], axis=1).dropna()
    df.columns = ["q", "p"]
 
    if len(df) < 5:
        stream_elas.append(np.nan)
        continue
 
    q_bar, p_bar = df["q"].mean(), df["p"].mean()
    dq = df["q"] - q_bar
    dp = df["p"] - p_bar
 
    elas = (dq/q_bar)/(dp/p_bar)
    stream_elas.append(np.nanmedian(elas))
 
# ======================================================
# HFD
# ======================================================
def calc_hfd(series):
    hfd = []
 
    for year, qy in series.groupby(series.index.year):
        total = qy.sum()
        if total <= 0:
            continue
 
        cum = qy.cumsum()
        hfd.append(cum[cum >= total/2].index[0].month)
 
    return pd.Series(hfd).mode().iloc[0] if len(hfd) else np.nan
 
hfd_mean = [calc_hfd(q_long[b]) for b in basins]
 
# ======================================================
# BASEFLOW INDEX
# ======================================================
def lyne_hollick(q, alpha=0.925):
    bf = np.zeros_like(q)
    for i in range(1, len(q)):
        bf[i] = alpha * bf[i-1] + (1+alpha)/2 * (q[i] - q[i-1])
        bf[i] = min(bf[i], q[i])
    bf[bf < 0] = 0
    return bf
 
baseflow_index = []
for b in basins:
    q = q_long[b].dropna().values
    if len(q) < 12:
        baseflow_index.append(np.nan)
        continue
    bf = lyne_hollick(q)
    baseflow_index.append(np.sum(bf)/np.sum(q))
 
# ======================================================
# FINAL DATAFRAME
# ======================================================
results = pd.DataFrame({
    "Basin_ID": basins,
    "q_mean": q_mean,
    "annual_q": annual_q,
    "runoff_ratio": runoff_ratio,
    "stream_elas": stream_elas,
    "hfd_mean": hfd_mean,
    "baseflow_index": baseflow_index,
    "slope_fdc": slope_fdc,
    "q_cv": q_cv,
    "Q5": q_5,
    "Q95": q_95,
    "Q_skew": q_skew,
    "high_q_freq": high_q_freq,
    "high_q_dur": high_q_dur,
    "low_q_freq": low_q_freq,
    "low_q_dur": low_q_dur,
    "zero_q_freq": zero_q_freq,
    "RLD": rld
})
 
# Add percentiles
results["Q10"] = q_10
results["Q25"] = q_25
results["Q50"] = q_50
results["Q75"] = q_75
results["Q90"] = q_90
 
# Monthly means
month_names = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
for i, m in enumerate(month_names):
    results[f"mean_{m}_flow"] = monthly_means[i+1]
 
# Merge area
#results = results.merge(area_df, on="Basin_ID", how="left")
 
#if "catchment_area" in results.columns:
#    results["runoff_per_km2"] = results["runoff_ratio"] / results["catchment_area"]
 
# ======================================================
# SAVE
# ======================================================
results.to_csv(out_fp, index=False)
 
print(f" DONE: {len(results.columns)} variables for {len(results)} basins")
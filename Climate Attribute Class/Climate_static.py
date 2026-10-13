import pandas as pd
import numpy as np

csv_path = ".../CAMELS_Afr_Climate_Daily_L2_p_mswep.csv"
df = pd.read_csv(csv_path)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Seasons
def get_season(m):
    if m in [12, 1, 2]:
        return "djf"
    elif m in [3, 4, 5]:
        return "mam"
    elif m in [6, 7, 8]:
        return "jja"
    else:
        return "son"

df["Season"] = df["Month"].apply(get_season)

print("Missing Dates:", df["Date"].isna().sum())

# IDENTIFY BASIN COLUMNS
basin_cols = [c for c in df.columns if c not in ["Date", "Year", "Month", "Season"]]
df[basin_cols] = df[basin_cols].apply(pd.to_numeric, errors="coerce")

def rle_lengths(arr):
    """Return lengths of consecutive True values"""
    lengths = []
    count = 0
    
    for val in arr:
        if val:
            count += 1
        else:
            if count > 0:
                lengths.append(count)
                count = 0
    if count > 0:
        lengths.append(count)
    
    return lengths

results = []

for basin in basin_cols:
    
    basin_df = df[["Date", "Year", "Month", "Season", basin]].dropna()
    
    if basin_df.empty:
        continue
    
    p = basin_df[basin].values
    years = basin_df["Year"].values
    
    n_years = len(np.unique(years))
    
    p_mean = np.nanmean(p)
    threshold_high = 5 * p_mean
    
    # HIGH PRECIP 
    high_days = p >= threshold_high
    high_prec_freq = np.sum(high_days) / n_years

    high_lengths = rle_lengths(high_days)
    high_prec_dur = np.mean(high_lengths) if len(high_lengths) > 0 else 0

    if np.sum(high_days) == 0:
        high_prec_timing = np.nan

    else:
        season_counts_total = basin_df["Season"].value_counts()
        
        high_season_df = basin_df.copy()
        high_season_df["high"] = high_days
        
        season_counts_high = high_season_df[high_season_df["high"]].groupby("Season").size()
        
        seasons = ["djf", "mam", "jja", "son"]
        season_counts_high = season_counts_high.reindex(seasons, fill_value=0)
        season_counts_total = season_counts_total.reindex(seasons, fill_value=0)
        
        season_ratio = season_counts_high / season_counts_total.replace(0, np.nan)
        
        max_val = season_ratio.max()
        top_seasons = season_ratio[season_ratio == max_val].index.tolist()
        
        if len(top_seasons) == 1:
            high_prec_timing = top_seasons[0]
        else:
            high_prec_timing = ",".join(top_seasons)
            print(f" Tie in HIGH precipitation timing for basin: {basin} → {top_seasons}")
    
    # LOW PRECIP
    low_days = p <= 1
    low_prec_freq = np.sum(low_days) / n_years

    low_lengths = rle_lengths(low_days)
    low_prec_dur = np.mean(low_lengths) if len(low_lengths) > 0 else 0

    if np.sum(low_days) == 0:
        low_prec_timing = np.nan

    else:
        low_season_df = basin_df.copy()
        low_season_df["low"] = low_days
        
        season_counts_low = low_season_df[low_season_df["low"]].groupby("Season").size()
        
        season_counts_low = season_counts_low.reindex(seasons, fill_value=0)
        
        season_ratio_low = season_counts_low / season_counts_total.replace(0, np.nan)
        
        max_val_low = season_ratio_low.max()
        top_seasons_low = season_ratio_low[season_ratio_low == max_val_low].index.tolist()
        
        if len(top_seasons_low) == 1:
            low_prec_timing = top_seasons_low[0]
        else:
            low_prec_timing = ",".join(top_seasons_low)
            print(f"Tie in LOW precipitation timing for basin: {basin} → {top_seasons_low}")
    
    # ANNUAL PRECIP
    annual_prec = basin_df.groupby("Year")[basin].sum().mean()
    
    # Monthly totals
    monthly_prec = basin_df.groupby("Month")[basin].sum()
    p_seasonality = monthly_prec.std() / monthly_prec.mean()
    
    # Annual mean spread
    annual_means = basin_df.groupby("Year")[basin].mean()
    p_mean_spread = annual_means.std() / annual_means.mean()
    
    results.append({
        "Basin_ID": basin,
        "high_prec_freq": high_prec_freq,
        "high_prec_dur": high_prec_dur,
        "high_prec_timing": high_prec_timing,
        "low_prec_freq": low_prec_freq,
        "low_prec_dur": low_prec_dur,
        "low_prec_timing": low_prec_timing,
        "p_seasonality": p_seasonality,
        "annual_prec": annual_prec,
        "p_mean_spread": p_mean_spread
    })

results_df = pd.DataFrame(results)

output_path = ".../CAMELS_Afr_Climate_Static_L2_mswep.csv"
results_df.to_csv(output_path, index=False)

print("Climate indices successfully computed and saved!")
print(results_df.head())

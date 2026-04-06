import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Re-load raw data  (same logic as step 1)
# ─────────────────────────────────────────────────────────────────────────────

# --- Pluviometer ---
pluvio_files = sorted(glob.glob("Pluviometer data/*.xlsx"))
pluvio_frames = []
for f in pluvio_files:
    df = pd.read_excel(f, header=1, usecols=[0, 1, 2])
    df.columns = ["datetime", "P01_mm", "P14_mm"]
    pluvio_frames.append(df)

rain_raw = pd.concat(pluvio_frames, ignore_index=True)
rain_raw["datetime"] = pd.to_datetime(rain_raw["datetime"], errors="coerce")
rain_raw = rain_raw.dropna(subset=["datetime"]).set_index("datetime").sort_index()
rain_raw["P01_mm"] = pd.to_numeric(rain_raw["P01_mm"], errors="coerce")
rain_raw["P14_mm"] = pd.to_numeric(rain_raw["P14_mm"], errors="coerce")

# --- Sewage ---
sewage_files = sorted(glob.glob("Sewage data/*.csv"))
sewage_frames = []
for f in sewage_files:
    df = pd.read_csv(f, parse_dates=["Date"])
    sewage_frames.append(df)

sewer_raw = pd.concat(sewage_frames, ignore_index=True)
sewer_raw = sewer_raw.rename(columns={"Date": "datetime", "Value": "level_mm"})
sewer_raw["datetime"] = pd.to_datetime(sewer_raw["datetime"])
sewer_raw = sewer_raw.set_index("datetime").sort_index()

print(f"Raw sewer : {sewer_raw.shape[0]:,} rows  ({sewer_raw.index.min()} to {sewer_raw.index.max()})")
print(f"Raw rain  : {rain_raw.shape[0]:,} rows  ({rain_raw.index.min()} to {rain_raw.index.max()})")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Clean sewer level (1-min data, BEFORE resampling)
# ─────────────────────────────────────────────────────────────────────────────

PHYSICAL_MIN  =    0.0   # mm — below 0 is impossible
PHYSICAL_MAX  = 3800.0   # mm — well above observed maximum (~3739 mm)
SPIKE_THRESH  =  500.0   # mm — jump in 1 min that flags a spike

sewer = sewer_raw.copy()

# Step 2a — physical bounds
n_neg  = (sewer["level_mm"] < PHYSICAL_MIN).sum()
n_ceil = (sewer["level_mm"] > PHYSICAL_MAX).sum()
sewer.loc[sewer["level_mm"] < PHYSICAL_MIN, "level_mm"] = np.nan
sewer.loc[sewer["level_mm"] > PHYSICAL_MAX, "level_mm"] = np.nan
print(f"\n[Sewer cleaning]")
print(f"  Removed {n_neg:,}  values below {PHYSICAL_MIN} mm (negatives)")
print(f"  Removed {n_ceil:,} values above {PHYSICAL_MAX} mm (hard ceiling)")

# Step 2b — spike detection on 1-min series
# A reading is a spike if it deviates > SPIKE_THRESH mm from BOTH its
# immediate predecessor AND its immediate successor.
level = sewer["level_mm"]
diff_forward  = (level - level.shift( 1)).abs()
diff_backward = (level - level.shift(-1)).abs()
spike_mask = (diff_forward > SPIKE_THRESH) & (diff_backward > SPIKE_THRESH)
n_spikes = spike_mask.sum()
sewer.loc[spike_mask, "level_mm"] = np.nan
print(f"  Removed {n_spikes:,} spike readings (|jump| > {SPIKE_THRESH} mm on both sides)")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Resample sewer from 1-min to 5-min (mean of clean readings)
# ─────────────────────────────────────────────────────────────────────────────

sewer_5min = (
    sewer["level_mm"]
    .resample("5min")
    .mean()          # NaN if all 5 readings in window were removed
    .rename("level_mm")
    .to_frame()
)

n_nan_after_resample = sewer_5min["level_mm"].isna().sum()
print(f"\n[Resample to 5-min]")
print(f"  Shape after resample : {sewer_5min.shape}")
print(f"  NaN bins after resample: {n_nan_after_resample:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Interpolate small gaps in sewer (up to 30 min = 6 steps at 5-min)
# ─────────────────────────────────────────────────────────────────────────────

MAX_INTERP_STEPS = 6   # 6 × 5 min = 30 min

# Track which timestamps are NaN before interpolation so we can report gap sizes
nan_mask_before = sewer_5min["level_mm"].isna()

sewer_5min["level_mm"] = sewer_5min["level_mm"].interpolate(
    method="time", limit=MAX_INTERP_STEPS, limit_direction="forward"
)

nan_mask_after = sewer_5min["level_mm"].isna()
n_filled  = (nan_mask_before & ~nan_mask_after).sum()
n_remain  = nan_mask_after.sum()
print(f"\n[Gap interpolation — sewer]")
print(f"  Interpolated : {n_filled:,} steps (gaps <= {MAX_INTERP_STEPS * 5} min)")
print(f"  Remaining NaN (long gaps): {n_remain:,} steps")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Clean rainfall — no negatives found, but clip defensively
# ─────────────────────────────────────────────────────────────────────────────

rain = rain_raw.copy()
n_neg_p01 = (rain["P01_mm"] < 0).sum()
n_neg_p14 = (rain["P14_mm"] < 0).sum()
rain.loc[rain["P01_mm"] < 0, "P01_mm"] = np.nan
rain.loc[rain["P14_mm"] < 0, "P14_mm"] = np.nan
print(f"\n[Rain cleaning]")
print(f"  Removed {n_neg_p01} negative P01 readings")
print(f"  Removed {n_neg_p14} negative P14 readings")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Merge on common 5-min index (inner join — only timestamps in both)
# ─────────────────────────────────────────────────────────────────────────────

df = sewer_5min.join(rain, how="inner")

print(f"\n[Merged dataset]")
print(f"  Shape      : {df.shape}")
print(f"  Date range : {df.index.min()}  to  {df.index.max()}")
print(f"  Columns    : {df.columns.tolist()}")
print(f"\nMissing values per column:")
print(df.isna().sum())
print(f"\nHead:")
print(df.head())
print(f"\nDescribe:")
print(df.describe().round(3))

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plots — full time series for visual QC  (interactive HTML via Plotly)
# ─────────────────────────────────────────────────────────────────────────────

# Downsample to hourly for the overview plot so it renders quickly in browser
df_plot = df.resample("1h").mean()

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=[
        "DO08 sewer level (5-min, hourly avg for overview)",
        "Rainfall — P01 Avant-Port",
        "Rainfall — P14 Flagey",
    ],
    vertical_spacing=0.07,
)

fig.add_trace(
    go.Scatter(x=df_plot.index, y=df_plot["level_mm"],
               mode="lines", line=dict(color="steelblue", width=1),
               name="DO08 level"),
    row=1, col=1,
)
fig.add_hline(y=3000, line_dash="dash", line_color="red",
              annotation_text="Overflow threshold (3000 mm)",
              annotation_position="top right", row=1, col=1)

fig.add_trace(
    go.Bar(x=df_plot.index, y=df_plot["P01_mm"],
           marker_color="cornflowerblue", name="P01 Avant-Port"),
    row=2, col=1,
)
fig.add_trace(
    go.Bar(x=df_plot.index, y=df_plot["P14_mm"],
           marker_color="mediumseagreen", name="P14 Flagey"),
    row=3, col=1,
)

fig.update_yaxes(title_text="Level (mm)", row=1, col=1)
fig.update_yaxes(title_text="Rain (mm/h)", row=2, col=1)
fig.update_yaxes(title_text="Rain (mm/h)", row=3, col=1)
fig.update_layout(height=800, title_text="Full time-series overview — cleaned data",
                  showlegend=True)

out_path = FIGURES_DIR + "step2_full_timeseries.html"
fig.write_html(out_path)
print(f"\nFull overview figure saved to: {out_path}")

# ── Zoom plot: 12-hour window around a representative overflow event ──────────
overflow_times = df.index[df["level_mm"] > 3000]
if len(overflow_times):
    event_center = overflow_times[len(overflow_times) // 2]
    window_start = event_center - pd.Timedelta(hours=6)
    window_end   = event_center + pd.Timedelta(hours=6)
    zoom = df.loc[window_start:window_end]

    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=[
            f"Sewer level — event centred on {event_center.strftime('%Y-%m-%d %H:%M')}",
            "Rainfall (5-min)",
        ],
        vertical_spacing=0.1,
    )
    fig2.add_trace(
        go.Scatter(x=zoom.index, y=zoom["level_mm"],
                   mode="lines", line=dict(color="steelblue", width=2),
                   name="DO08 level"),
        row=1, col=1,
    )
    fig2.add_hline(y=3000, line_dash="dash", line_color="red",
                   annotation_text="3000 mm", row=1, col=1)
    fig2.add_trace(
        go.Bar(x=zoom.index, y=zoom["P01_mm"], name="P01", marker_color="cornflowerblue"),
        row=2, col=1,
    )
    fig2.add_trace(
        go.Bar(x=zoom.index, y=zoom["P14_mm"], name="P14", marker_color="mediumseagreen"),
        row=2, col=1,
    )
    fig2.update_yaxes(title_text="Level (mm)", row=1, col=1)
    fig2.update_yaxes(title_text="Rain (mm/5min)", row=2, col=1)
    fig2.update_layout(height=600, barmode="overlay")

    out_path2 = FIGURES_DIR + "step2_overflow_zoom.html"
    fig2.write_html(out_path2)
    print(f"Overflow zoom figure saved to: {out_path2}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Save cleaned merged dataset for downstream steps
# ─────────────────────────────────────────────────────────────────────────────

df.to_parquet("data_clean.parquet")
print(f"\nClean dataset saved to: data_clean.parquet")
print(f"  Shape  : {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

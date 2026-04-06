import pandas as pd
import numpy as np
import glob

pd.set_option('future.no_silent_downcasting', True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Pluviometer data ──────────────────────────────────────────────────────────
pluvio_files  = sorted(glob.glob("Pluviometer data/*.xlsx"))
pluvio_frames = []

for f in pluvio_files:
    df = pd.read_excel(f, header=1, usecols=[0, 1, 2])
    df.columns = ["datetime", "Avant-Port_mm", "Flagey_mm"]
    pluvio_frames.append(df)
    print(f"Loaded: {f}  →  {df.shape}")

rain = pd.concat(pluvio_frames, ignore_index=True)
rain["datetime"]      = pd.to_datetime(rain["datetime"], errors="coerce")
rain                  = rain.dropna(subset=["datetime"])
rain                  = rain.set_index("datetime").sort_index()
rain["Avant-Port_mm"] = pd.to_numeric(rain["Avant-Port_mm"], errors="coerce")
rain["Flagey_mm"]     = pd.to_numeric(rain["Flagey_mm"],     errors="coerce")

print("\n" + "=" * 55)
print("PLUVIOMETER DATA")
print("=" * 55)
print(f"  Shape     : {rain.shape}")
print(f"  Date range: {rain.index.min()}  →  {rain.index.max()}")
print(f"  Frequency : {pd.infer_freq(rain.index[:200])}")
print(f"  Missing   :\n{rain.isna().sum()}")

# ── Sewage data ───────────────────────────────────────────────────────────────
sewage_files  = sorted(glob.glob("Sewage data/*.csv"))
sewage_frames = []

for f in sewage_files:
    df = pd.read_csv(f, parse_dates=["Date"])
    sewage_frames.append(df)
    print(f"Loaded: {f}  →  {df.shape}")

sewer = pd.concat(sewage_frames, ignore_index=True)
sewer = sewer.rename(columns={"Date": "datetime", "Value": "level_mm"})
sewer["datetime"] = pd.to_datetime(sewer["datetime"])
sewer = sewer.set_index("datetime").sort_index()

print("\n" + "=" * 55)
print("SEWAGE DATA  (DO08 — Lion overflow)")
print("=" * 55)
print(f"  Shape     : {sewer.shape}")
print(f"  Date range: {sewer.index.min()}  →  {sewer.index.max()}")
print(f"  Missing   :\n{sewer.isna().sum()}")
print(f"  Overflow readings (> 3000 mm): {(sewer['level_mm'] > 3000).sum():,}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN SEWER DATA
# ═══════════════════════════════════════════════════════════════════════════════

PHYSICAL_MIN     =    0.0   # mm — negative levels are impossible
SPIKE_THRESH     =  500.0   # mm — jump on both sides flags a spike
MAX_INTERP_STEPS =   10     # max consecutive NaNs to interpolate (10 × 1min = 10 min)
OVERFLOW_THRESHOLD = 3000   # mm — overflow definition

# ── 2a: Remove physically impossible values ───────────────────────────────────
sewer_clean = sewer.copy()
n_neg = (sewer_clean["level_mm"] < PHYSICAL_MIN).sum()
sewer_clean.loc[sewer_clean["level_mm"] < PHYSICAL_MIN, "level_mm"] = np.nan
print(f"\nRemoved {n_neg:,} negative readings")

# ── 2b: Spike detection ───────────────────────────────────────────────────────
level      = sewer_clean["level_mm"]
diff_fwd   = (level - level.shift( 1)).abs()
diff_bwd   = (level - level.shift(-1)).abs()
spike_mask = (diff_fwd > SPIKE_THRESH) & (diff_bwd > SPIKE_THRESH)
n_spikes   = spike_mask.sum()
sewer_clean.loc[spike_mask, "level_mm"] = np.nan
print(f"Removed {n_spikes:,} spike readings (jump > {SPIKE_THRESH} mm on both sides)")

# ── 2c: Classify NaN gaps ─────────────────────────────────────────────────────
is_nan       = sewer_clean["level_mm"].isna()
nan_group_id = (is_nan != is_nan.shift()).cumsum().where(is_nan)
block_sizes  = nan_group_id.map(nan_group_id.value_counts())

short_gap_mask = is_nan & (block_sizes <= MAX_INTERP_STEPS)
long_gap_mask  = is_nan & (block_sizes >  MAX_INTERP_STEPS)
n_short = short_gap_mask.sum()
n_long  = long_gap_mask.sum()

print(f"\nGap analysis:")
print(f"  Short gaps (≤ {MAX_INTERP_STEPS} min) : {n_short:,} readings → interpolated")
print(f"  Long  gaps (> {MAX_INTERP_STEPS} min) : {n_long:,}  readings → kept as NaN")

# ── 2d: Interpolate short gaps only ───────────────────────────────────────────
sewer_clean["level_mm"] = sewer_clean["level_mm"].interpolate(
    method="linear", limit=MAX_INTERP_STEPS
)

print(f"\n{'='*55}")
print(f"SEWER CLEANING SUMMARY")
print(f"{'='*55}")
print(f"  Original readings : {len(sewer):,}")
print(f"  Negatives removed : {n_neg:,}")
print(f"  Spikes removed    : {n_spikes:,}")
print(f"  Short gaps filled : {n_short:,}")
print(f"  Long gaps kept NaN: {n_long:,}")
print(f"  Final NaN count   : {sewer_clean['level_mm'].isna().sum():,}")

# ── 2e: Build overflow events table from raw 1-min cleaned data ───────────────
is_overflow  = sewer_clean["level_mm"] > OVERFLOW_THRESHOLD
event_id     = (is_overflow & ~is_overflow.shift(1).fillna(False)).cumsum()
event_id     = event_id.where(is_overflow)

events_raw = (
    sewer_clean.assign(event_id=event_id)
    .dropna(subset=["event_id"])
    .groupby("event_id")
    .agg(
        start    =("level_mm", lambda x: x.index.min()),
        end      =("level_mm", lambda x: x.index.max()),
        peak_mm  =("level_mm", "max"),
        mean_mm  =("level_mm", "mean"),
    )
    .reset_index(drop=True)
)
events_raw["duration_min"] = (
    (events_raw["end"] - events_raw["start"])
    .dt.total_seconds()
    .div(60)
    .add(1)
    .astype(int)
)
print(f"\nRaw overflow events detected: {len(events_raw)}")

# ── 2f: Gap sensitivity analysis — choose merging threshold ───────────────────
def count_events_for_gap(events_df, gap_min):
    gap     = pd.Timedelta(minutes=gap_min)
    df      = events_df.sort_values("start").reset_index(drop=True)
    merged  = []
    current = df.iloc[0].to_dict()
    for _, row in df.iloc[1:].iterrows():
        if row["start"] - current["end"] <= gap:
            current["end"]     = row["end"]
            current["peak_mm"] = max(current["peak_mm"], row["peak_mm"])
        else:
            merged.append(current)
            current = row.to_dict()
    merged.append(current)
    return len(merged)

print("\nGap sensitivity analysis:")
print("  Gap threshold → event count")
for g in [0, 5, 10, 15, 20, 30, 60]:
    print(f"    {g:>3} min  →  {count_events_for_gap(events_raw, g)} events")

# ── 2g: Merge events with gap < 10 min ───────────────────────────────────────
MIN_DRY_GAP    = pd.Timedelta(minutes=10)
merged_list    = []
current        = events_raw.iloc[0].to_dict()

for _, row in events_raw.iloc[1:].iterrows():
    if row["start"] - current["end"] <= MIN_DRY_GAP:
        current["end"]          = row["end"]
        current["peak_mm"]      = max(current["peak_mm"], row["peak_mm"])
        current["duration_min"] = int(
            (current["end"] - current["start"]).total_seconds() / 60
        ) + 1
    else:
        merged_list.append(current)
        current = row.to_dict()

merged_list.append(current)
events_merged = pd.DataFrame(merged_list).reset_index(drop=True)

# Recompute mean from raw signal for each merged event
def get_mean(start, end):
    mask = (
        (sewer_clean.index >= start) &
        (sewer_clean.index <= end)   &
        (sewer_clean["level_mm"] > OVERFLOW_THRESHOLD)
    )
    return sewer_clean.loc[mask, "level_mm"].mean()

events_merged["mean_mm"]  = events_merged.apply(
    lambda r: get_mean(r["start"], r["end"]), axis=1
)
events_merged["duration"] = events_merged["duration_min"].apply(
    lambda m: f"{m//60}h {m%60}min" if m >= 60 else f"{m}min"
)

print(f"\n{'='*55}")
print(f"OVERFLOW EVENT MERGING SUMMARY")
print(f"{'='*55}")
print(f"  Gap tolerance         : {MIN_DRY_GAP}")
print(f"  Events before merging : {len(events_raw)}")
print(f"  Events after merging  : {len(events_merged)}")
print(f"  Events merged away    : {len(events_raw) - len(events_merged)}")
print()
print(events_merged[["start", "end", "duration", "peak_mm", "mean_mm"]].to_string())

events_merged.to_csv("overflow_events_merged.csv", index=False)
print("\nSaved: overflow_events_merged.csv")
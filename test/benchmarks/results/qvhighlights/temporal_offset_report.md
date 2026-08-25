# Temporal Offset Analysis Report

**Queries analyzed:** 1542

## Aggregate Results

| Metric | Mean | |Mean| | Median | Std Dev | Min | Max |
|--------|------:|-------:|-------:|--------:|-----:|-----:|
| Start Offset (s) | +0.17 | 23.16 | -0.23 | 38.52 | -138.00 | +144.14 |
| End Offset (s) | -0.64 | 24.10 | +0.00 | 39.15 | -140.57 | +140.02 |
| Center Offset (s) | -0.23 | 19.99 | -0.33 | 32.74 | -129.32 | +141.52 |
| ABE (s) | +23.63 | 23.63 | +12.07 | 26.66 | +0.00 | +141.52 |

- **Mean IoU:** 0.4081
- **Median IoU:** 0.3740
- **Mean duration ratio (pred/gt):** 1.49
- **Complete misses (IoU=0):** 276 (17.9%)

## Breakdown by GT Moment Length

### Short (0-10s) — 177 queries

| Metric | Mean | |Mean| | Median |
|--------|------:|-------:|-------:|
| Start Offset | -3.74 | 18.34 | -1.60 |
| End Offset | +9.41 | 17.95 | +1.47 |
| Center Offset | +2.83 | 16.33 | -0.22 |
| ABE | +18.15 | 18.15 | +6.88 |
- IoU mean: 0.3409, Duration ratio mean: 3.20
- Complete misses: 38 (21.5%)

### Middle (10-30s) — 812 queries

| Metric | Mean | |Mean| | Median |
|--------|------:|-------:|-------:|
| Start Offset | -4.77 | 20.72 | -1.07 |
| End Offset | +5.34 | 21.26 | +0.26 |
| Center Offset | +0.28 | 18.54 | -0.14 |
| ABE | +20.99 | 20.99 | +8.74 |
- IoU mean: 0.4233, Duration ratio mean: 1.59
- Complete misses: 169 (20.8%)

### Long (30-150s) — 553 queries

| Metric | Mean | |Mean| | Median |
|--------|------:|-------:|-------:|
| Start Offset | +8.68 | 28.29 | +0.00 |
| End Offset | -12.63 | 30.23 | -1.87 |
| Center Offset | -1.98 | 23.30 | -0.93 |
| ABE | +29.26 | 29.26 | +20.59 |
- IoU mean: 0.4073, Duration ratio mean: 0.80
- Complete misses: 69 (12.5%)

## Interpretation Guide

| Pattern | Diagnosis |
|---------|-----------|
| Small |Center Offset|, large ABE | Right location, wrong boundaries (scene granularity) |
| Large |Center Offset| | Wrong part of the video (retrieval error) |
| Start Offset ~ 0, End Offset << 0 | Starts right but ends too early (scenes too short) |
| Duration ratio << 1.0 | Predicted clips much narrower than GT (granularity mismatch) |
| Duration ratio >> 1.0 | Predicted clips much wider than GT (over-merging) |

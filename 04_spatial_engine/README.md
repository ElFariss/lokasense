# 04 — Spatial Engine

## Purpose

Transforms timestamped signal predictions into geospatial opportunity scores and a browser-ready heatmap.

## What Is Implemented

- Time decay with a 30-day half-life: newer evidence weighs more than stale evidence.
- Franchise penalty inside the raw score before normalization.
- Score normalization against the actual formula bounds.
- Coordinate resolution through `LocationResolver`, using area hints, city hints, OpenStreetMap POIs, and any cached local review coordinates when available.
- Folium heatmap generation using resolved coordinates first and city-level fallback only when no better location is available.

## Directory Structure

```text
04_spatial_engine/
├── README.md
└── modelling/
    ├── scoring.py
    └── heatmap.py
```

## Opportunity Scoring Formula

```text
raw_signal_score =
    0.30 * DEMAND_UNMET
  + 0.15 * DEMAND_PRESENT
  + 0.10 * TREND
  - 0.20 * COMPETITION_HIGH
  - 0.10 * COMPLAINT
  - 0.05 * SUPPLY_SIGNAL

raw_score_after_penalty = raw_signal_score - 0.10 * franchise_ratio
normalized_score = clamp((raw_score_after_penalty - min_raw) / (max_raw - min_raw), 0, 1)
```

`min_raw` and `max_raw` are derived from the actual scoring bounds instead of an ad hoc shift.

## Output Fields

`opportunity_scores.csv` now includes:

- `opportunity_score`
- `raw_signal_score`
- `raw_score_after_penalty`
- `franchise_ratio`
- `avg_age_days`
- `avg_decay_weight`
- `resolved_lat`
- `resolved_lng`
- `resolution_source`

## Heatmap Colors

| Color | Score Range | Meaning |
|-------|-------------|---------|
| Green | `>= 0.65` | Strong opportunity |
| Yellow | `0.40–0.65` | Moderate opportunity |
| Red | `< 0.40` | Saturated or risky |

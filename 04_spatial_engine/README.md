# 04 — Spatial Engine

## Purpose
Transforms model predictions into geospatial opportunity maps.

## Directory Structure
```
04_spatial_engine/
├── README.md            ← This file
├── dataset/
│   └── gazetteer.py     ← Load admin boundary data + GeoJSON
├── processing/
│   ├── location_expand.py ← Expand city → kecamatan list
│   └── aggregate.py      ← Spatial aggregation per area
└── modelling/
    ├── scoring.py        ← Opportunity scoring formula
    └── heatmap.py        ← Folium choropleth map generation
```

## Opportunity Scoring Formula
```
score = w₁·unmet + w₂·present + w₃·trend − w₄·competition − w₅·complaint − w₆·supply − w₇·franchise_ratio
```
Default weights: w₁=0.30, w₂=0.15, w₃=0.10, w₄=0.20, w₅=0.10, w₆=0.05, w₇=0.10

## Heatmap Color Coding
| Color | Score Range | Meaning |
|-------|------------|---------|
| 🟢 Green | ≥0.65 | Strong opportunity |
| 🟡 Yellow | 0.40–0.65 | Moderate opportunity |
| 🔴 Red | <0.40 | Saturated/risky |
| ⬜ Gray | N/A | Insufficient data |

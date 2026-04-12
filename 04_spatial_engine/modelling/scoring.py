#!/usr/bin/env python3
"""
04_spatial_engine/modelling/scoring.py
Compute opportunity scores per (kecamatan × business_type).
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.location_resolution import LocationResolver

DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Default scoring weights
WEIGHTS = {
    "DEMAND_UNMET": 0.30,
    "DEMAND_PRESENT": 0.15,
    "TREND": 0.10,
    "COMPETITION_HIGH": -0.20,
    "COMPLAINT": -0.10,
    "SUPPLY_SIGNAL": -0.05,
    "NEUTRAL": 0.0,
}
HALF_LIFE_DAYS = 30.0
DECAY_LAMBDA = np.log(2) / HALF_LIFE_DAYS
MIN_SIGNAL_SCORE = min(WEIGHTS.values())
MAX_SIGNAL_SCORE = max(WEIGHTS.values())
MIN_RAW_SCORE = MIN_SIGNAL_SCORE - 0.10
MAX_RAW_SCORE = MAX_SIGNAL_SCORE


def compute_decay_weights(group: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    timestamps = pd.to_datetime(group.get("timestamp"), utc=True, errors="coerce")
    if timestamps.isna().all():
        age_days = pd.Series(np.zeros(len(group)), index=group.index, dtype=float)
        decay = pd.Series(np.ones(len(group)), index=group.index, dtype=float)
        return age_days, decay

    now = pd.Timestamp.utcnow()
    age_days = ((now - timestamps).dt.total_seconds() / 86400.0).clip(lower=0).fillna(0.0)
    decay = np.exp(-DECAY_LAMBDA * age_days)
    return age_days, pd.Series(decay, index=group.index, dtype=float)


def normalize_score(raw_score: float) -> float:
    scaled = (raw_score - MIN_RAW_SCORE) / (MAX_RAW_SCORE - MIN_RAW_SCORE)
    return float(max(0.0, min(1.0, scaled)))


def compute_opportunity_scores(labeled_df, poi_df=None, resolver: LocationResolver | None = None):
    """
    Compute opportunity score per (city, kecamatan, business_type).
    
    Score = Σ(weight_signal × count_signal / total) - franchise_penalty
    """
    results = []
    resolver = resolver or LocationResolver()
    
    # Group by area + business type
    if 'area_hint' not in labeled_df.columns:
        labeled_df['area_hint'] = 'unknown'
    if 'city' not in labeled_df.columns:
        labeled_df['city'] = 'unknown'
    if 'business_hint' not in labeled_df.columns:
        labeled_df['business_hint'] = 'general'

    groups = labeled_df.groupby(['city', 'area_hint', 'business_hint'])

    for (city, kec, biz), group in groups:
        total = len(group)
        if total < 3:
            continue

        # Count signals
        signal_col = 'final_signal' if 'final_signal' in group.columns else 'signal'
        signal_counts = group[signal_col].value_counts().to_dict()
        age_days, decay_weights = compute_decay_weights(group)
        total_weight = float(decay_weights.sum()) or 1.0

        # Compute weighted score
        signal_score = 0.0
        for signal, weight in WEIGHTS.items():
            signal_weight = float(decay_weights[group[signal_col] == signal].sum())
            rate = signal_weight / total_weight
            signal_score += weight * rate

        # Franchise penalty from POI data
        franchise_ratio = 0.0
        if poi_df is not None and len(poi_df) > 0:
            city_pois = poi_df[(poi_df['city'] == city) & (poi_df['business_type'] == biz)]
            if len(city_pois) > 0:
                franchise_ratio = city_pois['is_franchise'].mean()

        raw_score = signal_score - (0.10 * franchise_ratio)
        score = normalize_score(raw_score)
        location = resolver.resolve_area(city, kec)

        # Color coding
        if score >= 0.65:
            color = "green"
            label = "Strong Opportunity"
        elif score >= 0.40:
            color = "yellow"
            label = "Moderate Opportunity"
        else:
            color = "red"
            label = "Saturated / Risky"

        results.append({
            "city": city,
            "kecamatan": kec,
            "business_type": biz,
            "opportunity_score": round(score, 3),
            "color": color,
            "label": label,
            "total_signals": total,
            "signal_breakdown": signal_counts,
            "raw_signal_score": round(signal_score, 4),
            "raw_score_after_penalty": round(raw_score, 4),
            "franchise_ratio": round(franchise_ratio, 3),
            "avg_age_days": round(float(age_days.mean()), 2),
            "avg_decay_weight": round(float(decay_weights.mean()), 4),
            "resolved_lat": location["lat"],
            "resolved_lng": location["lng"],
            "resolution_source": location["resolution_source"],
        })

    results_df = pd.DataFrame(results)
    
    # Save
    results_df.to_csv(LOG_DIR / "opportunity_scores.csv", index=False)
    
    with open(LOG_DIR / "opportunity_scores.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Scored {len(results)} area x business combinations")
    return results_df


def main():
    print("=" * 60)
    print(" Opportunity Scoring — LokaSense")
    print("=" * 60)

    # Load labeled data
    labeled_file = DATA_DIR / "labeled" / "weak_labeled.csv"
    if not labeled_file.exists():
        print("No labeled data found.")
        return

    labeled_df = pd.read_csv(labeled_file)
    
    # Load POI data if available
    poi_file = DATA_DIR / "poi" / "overpass_poi.csv"
    poi_df = pd.read_csv(poi_file) if poi_file.exists() else None

    scores_df = compute_opportunity_scores(labeled_df, poi_df)
    
    if len(scores_df) > 0:
        print(f"\nScore Distribution:")
        print(scores_df[['city', 'kecamatan', 'business_type', 'opportunity_score', 'label']].to_string(index=False))


if __name__ == "__main__":
    main()

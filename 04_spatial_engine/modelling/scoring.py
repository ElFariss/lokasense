#!/usr/bin/env python3
"""
04_spatial_engine/modelling/scoring.py
Compute opportunity scores per (kecamatan × business_type).
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent.parent
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


def compute_opportunity_scores(labeled_df, poi_df=None):
    """
    Compute opportunity score per (city, kecamatan, business_type).
    
    Score = Σ(weight_signal × count_signal / total) - franchise_penalty
    """
    results = []
    
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

        # Compute weighted score
        score = 0.0
        for signal, weight in WEIGHTS.items():
            count = signal_counts.get(signal, 0)
            rate = count / total
            score += weight * rate

        # Normalize to 0-1 range
        score = max(0, min(1, (score + 0.5)))  # shift from [-0.5, 0.5] to [0, 1]

        # Franchise penalty from POI data
        franchise_ratio = 0.0
        if poi_df is not None and len(poi_df) > 0:
            city_pois = poi_df[(poi_df['city'] == city) & (poi_df['business_type'] == biz)]
            if len(city_pois) > 0:
                franchise_ratio = city_pois['is_franchise'].mean()
                score -= 0.10 * franchise_ratio

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
            "franchise_ratio": round(franchise_ratio, 3),
        })

    results_df = pd.DataFrame(results)
    
    # Save
    results_df.to_csv(LOG_DIR / "opportunity_scores.csv", index=False)
    
    with open(LOG_DIR / "opportunity_scores.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Scored {len(results)} area × business combinations")
    return results_df


def main():
    print("=" * 60)
    print(" Opportunity Scoring — LokaSense")
    print("=" * 60)

    # Load labeled data
    labeled_file = DATA_DIR / "labeled" / "weak_labeled.csv"
    if not labeled_file.exists():
        print("⚠ No labeled data found!")
        return

    labeled_df = pd.read_csv(labeled_file)
    
    # Load POI data if available
    poi_file = DATA_DIR / "poi" / "overpass_poi.csv"
    poi_df = pd.read_csv(poi_file) if poi_file.exists() else None

    scores_df = compute_opportunity_scores(labeled_df, poi_df)
    
    if len(scores_df) > 0:
        print(f"\n📊 Score Distribution:")
        print(scores_df[['city', 'kecamatan', 'business_type', 'opportunity_score', 'label']].to_string(index=False))


if __name__ == "__main__":
    main()

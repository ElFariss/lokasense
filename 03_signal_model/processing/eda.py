#!/usr/bin/env python3
"""
03_signal_model/processing/eda.py
Exploratory Data Analysis for the signal classification dataset.
Saves metrics and plots to logs/ for reproducibility.
"""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_LABELS = ["NEUTRAL", "DEMAND_UNMET", "DEMAND_PRESENT", "SUPPLY_SIGNAL", "COMPETITION_HIGH", "COMPLAINT", "TREND"]


def eda_smsa():
    """EDA on SmSA dataset."""
    smsa_dir = DATA_DIR / "huggingface" / "smsa"
    train_f = smsa_dir / "train.csv"
    if not train_f.exists():
        print("  SmSA not downloaded yet")
        return {}
    df = pd.read_csv(train_f)
    stats = {
        "dataset": "SmSA",
        "total_samples": len(df),
        "columns": list(df.columns),
        "text_len_mean": float(df.iloc[:, 0].astype(str).str.len().mean()),
        "text_len_std": float(df.iloc[:, 0].astype(str).str.len().std()),
        "label_distribution": dict(Counter(df.iloc[:, -1].astype(str).tolist())),
    }
    print(f"  SmSA: {stats['total_samples']} samples, avg text len {stats['text_len_mean']:.0f}")
    return stats


def eda_nusax():
    """EDA on NusaX sentiment."""
    nusax_dir = DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian"
    files = list(nusax_dir.glob("*.csv"))
    if not files:
        print("  NusaX not found")
        return {}
    all_texts = []
    for f in files:
        df = pd.read_csv(f)
        all_texts.append(df)
    combined = pd.concat(all_texts, ignore_index=True)
    stats = {
        "dataset": "NusaX",
        "total_samples": len(combined),
        "columns": list(combined.columns),
        "text_len_mean": float(combined.iloc[:, 0].astype(str).str.len().mean()),
        "label_distribution": dict(Counter(combined.iloc[:, -1].astype(str).tolist())),
    }
    print(f"  NusaX: {stats['total_samples']} samples")
    return stats


def eda_weak_labels():
    """EDA on weak-labeled data."""
    weak_f = DATA_DIR / "labeled" / "weak_labeled.csv"
    if not weak_f.exists():
        print("  Weak labels not generated yet")
        return {}
    df = pd.read_csv(weak_f)
    signal_counts = dict(Counter(df['signal'].tolist()))
    confidence_stats = {
        "mean": float(df['confidence'].mean()),
        "std": float(df['confidence'].std()),
        "high_conf_pct": float((df['confidence'] >= 0.8).mean() * 100),
    }
    stats = {
        "dataset": "weak_labeled",
        "total_samples": len(df),
        "signal_distribution": signal_counts,
        "confidence": confidence_stats,
        "source_distribution": dict(Counter(df['source'].tolist())),
    }

    # Plot signal distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Signal classes
    signals = pd.Series(signal_counts).sort_values(ascending=True)
    signals.plot(kind='barh', ax=axes[0], color=sns.color_palette("Set2", len(signals)))
    axes[0].set_title("Signal Class Distribution (Weak Labels)")
    axes[0].set_xlabel("Count")
    
    # Confidence histogram
    df['confidence'].plot(kind='hist', bins=20, ax=axes[1], color='steelblue', edgecolor='white')
    axes[1].set_title("Label Confidence Distribution")
    axes[1].set_xlabel("Confidence")
    
    plt.tight_layout()
    plt.savefig(LOG_DIR / "eda_signal_distribution.png", dpi=150)
    plt.close()
    print(f"  Weak labels: {len(df)} samples, {confidence_stats['high_conf_pct']:.1f}% high confidence")
    print(f"  Plot saved: logs/eda_signal_distribution.png")
    return stats


def eda_gmaps():
    """EDA on Google Maps reviews."""
    gmaps_f = DATA_DIR / "social_media" / "gmaps_reviews.csv"
    if not gmaps_f.exists():
        print("  Google Maps reviews not collected yet")
        return {}
    df = pd.read_csv(gmaps_f)
    stats = {
        "dataset": "gmaps_reviews",
        "total_reviews": len(df),
        "cities": dict(Counter(df['city'].tolist())),
        "business_types": dict(Counter(df['business_hint'].tolist())),
        "avg_text_len": float(df['text'].str.len().mean()),
        "avg_review_rating": float(df['review_rating'].mean()) if 'review_rating' in df.columns else None,
    }
    print(f"  Google Maps: {len(df)} reviews across {len(stats['cities'])} cities")
    return stats


def main():
    print("=" * 60)
    print(" Exploratory Data Analysis — LokaSense")
    print("=" * 60)

    all_stats = {}
    
    print("\nSmSA (IndoNLU):")
    all_stats["smsa"] = eda_smsa()
    
    print("\nNusaX Sentiment:")
    all_stats["nusax"] = eda_nusax()
    
    print("\nWeak-Labeled Data:")
    all_stats["weak_labels"] = eda_weak_labels()
    
    print("\nGoogle Maps Reviews:")
    all_stats["gmaps"] = eda_gmaps()

    # Save all EDA stats
    eda_file = LOG_DIR / "eda_stats.json"
    with open(eda_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nAll EDA stats saved to {eda_file}")


if __name__ == "__main__":
    main()

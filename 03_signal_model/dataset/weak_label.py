#!/usr/bin/env python3
"""
Weak labeling pipeline for 7-class market signal classification.
Applies keyword/regex rules to assign initial signal labels to unlabeled text data.

This is a pre-training step — the weak labels will be refined by:
1. Gemini pseudo-labeling (gemini_pseudo_label.py)
2. Human review (manual annotation)

Signal Classes:
    DEMAND_UNMET     — Unmet demand = strongest opportunity signal
    DEMAND_PRESENT   — Validates existing demand
    SUPPLY_SIGNAL    — Factual supply/saturation detection
    COMPETITION_HIGH — Subjective saturation perception
    COMPLAINT        — Differentiation opportunity
    TREND            — Timing/viral window
    NEUTRAL          — No actionable signal

Usage:
    python scripts/weak_labeling.py
"""
import re
import csv
import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "labeled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Weak Labeling Rules ───────────────────────────────────────────────────
# Each rule is a list of (pattern, weight) tuples
# Higher weight = stronger signal for that class

SIGNAL_RULES = {
    "DEMAND_UNMET": {
        "keywords": [
            # Core unmet demand
            "ga ada di", "gak ada di", "gak ada", "belum ada",
            "bikin sendiri", "pengen banget", "kapan buka",
            "kangen", "rindu", "mau dong", "butuh",
            "tolong buka", "mohon buka", "please buka",
            "pengennya ada", "andai ada", "seandainya ada",
            "susah cari", "susah nyari", "susah dapet",
            # Demand with location specificity
            "di sini ga ada", "di sini belum ada",
            "di kota ini belum", "daerah sini ga ada",
            "ga jual", "gak jual", "belum jual",
            "ga bisa dapet", "ga bisa beli",
        ],
        "negative_keywords": ["ada dong", "udah ada", "sudah ada"],
    },
    "DEMAND_PRESENT": {
        "keywords": [
            # Positive demand validation
            "enak banget", "enak bgt", "enaaak",
            "murah meriah", "murah banget", "affordable",
            "recommended", "rekomen", "rekomendasi",
            "wajib coba", "harus coba", "must try",
            "favorit", "favourite", "favorite",
            "tempat favorit", "langganan", "repeat order",
            "worth it", "worthit", "mantap",
            "puas", "satisfied", "puaaas",
            "best", "terbaik", "juara", "juaaraa",
            "suka banget", "love", "cinta",
        ],
        "negative_keywords": [],
    },
    "SUPPLY_SIGNAL": {
        "keywords": [
            # Factual supply observations
            "udah ada", "sudah ada", "ada banyak",
            "banyak yang jual", "banyak yg jual",
            "penuh warung", "penuh toko",
            "menjamur", "bertumbuh",
            "cabang baru", "outlet baru", "buka cabang",
            "ada 2", "ada 3", "ada 4", "ada 5",
            "ada dua", "ada tiga", "ada empat",
            # Specific supply info
            "total ada", "jumlahnya", "hitungannya",
            "satu jalan ada", "berjajar",
        ],
        "negative_keywords": [],
    },
    "COMPETITION_HIGH": {
        "keywords": [
            # Subjective saturation
            "banyak banget", "banyak bgt", "banyaaak",
            "dimana-mana", "di mana-mana", "dimana mana",
            "udah banyak", "udah bnyk", "udah kebanyakan",
            "terlalu banyak", "kebanyakan",
            "saingan", "kompetitor", "persaingan ketat",
            "saturated", "jenuh", "pasar jenuh",
            "ramai banget", "rame bgt",
            "semua jualan", "semua jual",
            "kayak jamur", "kaya jamur",
        ],
        "negative_keywords": [],
    },
    "COMPLAINT": {
        "keywords": [
            # Direct complaints
            "mahal banget", "mahal bgt", "kemahalan",
            "jelek", "mengecewakan", "kecewa",
            "kotor", "jorok", "dekil",
            "lama banget", "lama bgt", "kelamaan",
            "rasanya b aja", "biasa aja", "nothing special",
            "ga enak", "gak enak", "tidak enak",
            "pelayanan buruk", "service buruk",
            "porsi kecil", "porsinya kecil",
            "overpriced", "overprice", "ga worth",
            "kapok", "kecewain", "disappointed",
            "jangan kesini", "jangan ke sini",
            "bintang 1", "1 star",
        ],
        "negative_keywords": [],
    },
    "TREND": {
        "keywords": [
            # Viral/trending signals
            "viral", "viralll", "viraaaal",
            "lagi hits", "lagi hype", "lagi trend",
            "trending", "trendy",
            "FYP", "fyp", "for you page",
            "wajib dicoba", "must visit",
            "baru buka", "grand opening", "opening",
            "antre panjang", "antri panjang", "ngantri",
            "rame", "ramai", "ramee",
            "penasaran", "bikin penasaran",
            "tiktok", "tt viral",
        ],
        "negative_keywords": ["udah ga viral", "ga viral lagi"],
    },
}


def classify_text(text: str) -> tuple:
    """
    Classify text into market signal using keyword matching.
    Returns (signal_class, confidence, matched_keywords).

    Confidence scale:
        1.0: ≥3 keyword matches
        0.8: 2 keyword matches
        0.6: 1 keyword match with strong signal
        0.0: No match → NEUTRAL
    """
    text_lower = text.lower()

    scores = {}
    matches = {}

    for signal, rules in SIGNAL_RULES.items():
        # Check negative keywords first (exclusions)
        excluded = False
        for neg_kw in rules.get("negative_keywords", []):
            if neg_kw in text_lower:
                excluded = True
                break

        if excluded:
            continue

        # Count keyword matches
        matched = []
        for kw in rules["keywords"]:
            if kw in text_lower:
                matched.append(kw)

        if matched:
            scores[signal] = len(matched)
            matches[signal] = matched

    if not scores:
        return "NEUTRAL", 0.0, []

    # Get the signal with the most keyword matches
    best_signal = max(scores, key=scores.get)
    match_count = scores[best_signal]

    if match_count >= 3:
        confidence = 1.0
    elif match_count == 2:
        confidence = 0.8
    else:
        confidence = 0.6

    return best_signal, confidence, matches[best_signal]


def load_all_raw_texts() -> list:
    """
    Load all available text data from various sources.
    Returns list of dicts with 'text', 'source', 'timestamp', 'area_hint', 'business_hint'.
    """
    texts = []

    # 1. Load SmSA data (if downloaded)
    smsa_dir = DATA_DIR / "huggingface" / "smsa"
    if (smsa_dir / "train.csv").exists():
        for split in ["train", "validation", "test"]:
            fpath = smsa_dir / f"{split}.csv"
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = row.get("text", row.get("sentence", "")).strip()
                        if text and len(text) > 15:
                            texts.append({
                                "text": text,
                                "source": "smsa",
                                "timestamp": "",
                                "area_hint": "",
                                "business_hint": "",
                                "original_label": row.get("label", ""),
                            })
        print(f"  Loaded SmSA: {sum(1 for t in texts if t['source'] == 'smsa')} texts")

    # 2. Load NusaX sentiment data
    nusax_dir = DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian"
    for split in ["train", "valid", "test"]:
        fpath = nusax_dir / f"{split}.csv"
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "").strip()
                    if text and len(text) > 15:
                        texts.append({
                            "text": text,
                            "source": "nusax",
                            "timestamp": "",
                            "area_hint": "",
                            "business_hint": "",
                            "original_label": row.get("label", ""),
                        })
    print(f"  Loaded NusaX: {sum(1 for t in texts if t['source'] == 'nusax')} texts")

    # 3. Load Indonesian Sentiment data
    indosent_dir = DATA_DIR / "huggingface" / "indonesian_sentiment"
    if (indosent_dir / "train.csv").exists():
        for split in ["train", "validation", "test"]:
            fpath = indosent_dir / f"{split}.csv"
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = row.get("text", row.get("sentence", "")).strip()
                        if text and len(text) > 15:
                            texts.append({
                                "text": text,
                                "source": "indonesian_sentiment",
                                "timestamp": "",
                                "area_hint": "",
                                "business_hint": "",
                                "original_label": row.get("label", ""),
                            })
        print(f"  Loaded Indonesian Sentiment: {sum(1 for t in texts if t['source'] == 'indonesian_sentiment')} texts")

    # 4. Load Google Maps reviews (if collected)
    gmaps_file = DATA_DIR / "social_media" / "gmaps_reviews.csv"
    if gmaps_file.exists():
        with open(gmaps_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                if text and len(text) > 15:
                    texts.append({
                        "text": text,
                        "source": "google_maps",
                        "timestamp": row.get("timestamp", ""),
                        "area_hint": row.get("area_hint", ""),
                        "business_hint": row.get("business_hint", ""),
                        "original_label": "",
                    })
        print(f"  Loaded Google Maps: {sum(1 for t in texts if t['source'] == 'google_maps')} texts")

    # 5. Load TikTok data (if collected)
    tiktok_file = DATA_DIR / "social_media" / "tiktok_data.csv"
    if tiktok_file.exists():
        with open(tiktok_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                if text and len(text) > 15:
                    texts.append({
                        "text": text,
                        "source": "tiktok",
                        "timestamp": row.get("timestamp", ""),
                        "area_hint": row.get("area_hint", ""),
                        "business_hint": row.get("business_hint", ""),
                        "original_label": "",
                    })
        print(f"  Loaded TikTok: {sum(1 for t in texts if t['source'] == 'tiktok')} texts")

    return texts


def main():
    print("=" * 70)
    print(" Weak Labeling Pipeline — 7-Class Market Signal")
    print("=" * 70)

    # Load all raw texts
    print("\n📂 Loading raw text data...")
    texts = load_all_raw_texts()
    print(f"\n  Total texts loaded: {len(texts)}")

    if not texts:
        print("\n⚠ No texts found! Run setup_datasets.py and/or collection scripts first.")
        return

    # Apply weak labeling
    print("\n🏷️  Applying weak labeling rules...")
    labeled = []
    for t in texts:
        signal, confidence, matched_kws = classify_text(t["text"])
        labeled.append({
            "text": t["text"],
            "signal": signal,
            "confidence": confidence,
            "matched_keywords": "|".join(matched_kws),
            "source": t["source"],
            "timestamp": t["timestamp"],
            "area_hint": t["area_hint"],
            "business_hint": t["business_hint"],
            "original_label": t.get("original_label", ""),
        })

    # Statistics
    signal_counts = Counter(item["signal"] for item in labeled)
    print(f"\n📊 Signal Distribution:")
    total = len(labeled)
    for signal, count in signal_counts.most_common():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {signal:20s}  {count:>6}  ({pct:5.1f}%)  {bar}")

    high_conf = sum(1 for item in labeled if item["confidence"] >= 0.8)
    print(f"\n  High confidence (≥0.8): {high_conf} ({high_conf / total * 100:.1f}%)")
    print(f"  Needs review: {total - high_conf} ({(total - high_conf) / total * 100:.1f}%)")

    # Save
    output_file = OUTPUT_DIR / "weak_labeled.csv"
    fieldnames = labeled[0].keys()
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)

    print(f"\n✅ Saved {len(labeled)} labeled texts to {output_file}")

    # Save summary for manual review
    summary_file = OUTPUT_DIR / "weak_label_summary.json"
    summary = {
        "total_texts": total,
        "signal_distribution": dict(signal_counts.most_common()),
        "high_confidence_count": high_conf,
        "sources": dict(Counter(item["source"] for item in labeled).most_common()),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

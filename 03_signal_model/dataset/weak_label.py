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
import sys
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.text_normalization import normalize_for_dedupe

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
            "belumada", "tolongbuka",
            "pengennya ada", "andai ada", "seandainya ada",
            "susah cari", "susah nyari", "susah dapet",
            "lagi cari", "nyari", "butuh tempat", "kurang pilihan",
            "belum nemu", "belum ketemu", "minta buka", "harusnya ada",
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
            "wajibcoba", "haruscoba",
            "favorit", "favourite", "favorite",
            "tempat favorit", "langganan", "repeat order",
            "worth it", "worthit", "mantap",
            "puas", "satisfied", "puaaas",
            "best", "terbaik", "juara", "juaaraa",
            "suka banget", "love", "cinta",
            "langsung laris", "ramai pembeli", "rame pembeli",
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
            "cabangbaru", "bukacabang",
            "ada 2", "ada 3", "ada 4", "ada 5",
            "ada dua", "ada tiga", "ada empat",
            # Specific supply info
            "total ada", "jumlahnya", "hitungannya",
            "satu jalan ada", "berjajar",
            "sudah tersedia", "udah tersedia", "tersedia",
            "cabangnya", "cabang banyak", "punya cabang",
            "tempat baru terus", "buka terus", "lagi ekspansi",
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
            "sainganbanyak",
            "saturated", "jenuh", "pasar jenuh",
            "ramai banget", "rame bgt",
            "semua jualan", "semua jual",
            "kayak jamur", "kaya jamur",
            "saingannya banyak", "banyak pilihan",
            "hampir tiap jalan", "tiap jalan ada", "tempat baru terus",
            "cafe baru terus", "warung baru terus", "kedai baru terus",
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
            "pelayanan lama", "kurang enak", "ga worth it",
            "gak worth it", "rasanya biasa", "rasa biasa aja",
        ],
        "negative_keywords": [],
    },
    "TREND": {
        "keywords": [
            # Viral/trending signals
            "viral", "viralll", "viraaaal",
            "lagi hits", "lagi hype", "lagi trend",
            "lagihits", "lagiviral", "bukabaru", "grandopening", "antripanjang",
            "trending", "trendy",
            "FYP", "fyp", "for you page",
            "wajib dicoba", "must visit",
            "baru buka", "grand opening", "opening",
            "antre panjang", "antri panjang", "ngantri",
            "rame", "ramai", "ramee",
            "penasaran", "bikin penasaran",
            "tiktok", "tt viral",
            "lagi rame", "rame terus", "ramai terus", "lagi ramai",
        ],
        "negative_keywords": ["udah ga viral", "ga viral lagi"],
    },
}


def classify_text(text: str, context_text: str = "") -> tuple:
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
    context_lower = context_text.lower()

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
        context_matched = []
        for kw in rules["keywords"]:
            if kw in text_lower:
                matched.append(kw)
            elif context_lower and kw in context_lower:
                context_matched.append(f"query:{kw}")

        score = len(matched) + (0.75 * len(context_matched))
        if score > 0:
            scores[signal] = score
            matches[signal] = matched + context_matched

    if not scores:
        return "NEUTRAL", 0.0, []

    # Get the signal with the most keyword matches
    best_signal = max(scores, key=scores.get)
    match_count = scores[best_signal]

    if match_count >= 2.5:
        confidence = 1.0
    elif match_count >= 1.5:
        confidence = 0.8
    else:
        confidence = 0.6

    return best_signal, confidence, matches[best_signal]


def load_all_raw_texts() -> list:
    """
    Load all available text data from the active scraped bootstrap path.
    Returns list of dicts with 'text', 'source', 'timestamp', 'area_hint', 'business_hint'.
    """
    texts = []
    seen_hashes = set()

    def add_text(row: dict, default_source: str) -> None:
        text = (row.get("text") or row.get("raw_text") or "").strip()
        if len(text) <= 15:
            return
        dedupe_key = normalize_for_dedupe(text)
        if not dedupe_key:
            return
        dedupe_hash = dedupe_key
        if dedupe_hash in seen_hashes:
            return
        seen_hashes.add(dedupe_hash)
        texts.append({
            "text": text,
            "source": row.get("source", default_source),
            "timestamp": row.get("timestamp", ""),
            "area_hint": row.get("area_hint", ""),
            "business_hint": row.get("business_hint", ""),
            "query": row.get("query", ""),
            "original_label": row.get("original_label", ""),
        })

    bootstrap_file = DATA_DIR / "scraped" / "signal_bootstrap.csv"
    if bootstrap_file.exists():
        with open(bootstrap_file, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                add_text(row, row.get("platform", "scraped_bootstrap"))
        print(f"  Loaded scraped bootstrap: {len(texts)} texts")
        return texts

    social_files = [
        (DATA_DIR / "social_media" / "tiktok_data.csv", "tiktok"),
        (DATA_DIR / "social_media" / "instagram_data.csv", "instagram"),
        (DATA_DIR / "social_media" / "x_data.csv", "x"),
        (DATA_DIR / "social_media" / "gmaps_reviews.csv", "google_maps"),
    ]

    for filepath, source_name in social_files:
        if not filepath.exists():
            continue
        with open(filepath, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                add_text(row, source_name)
        print(f"  Loaded {source_name}: {sum(1 for text in texts if text['source'] == source_name or text['source'].startswith(source_name))} texts")

    return texts


def main():
    print("=" * 70)
    print(" Weak Labeling Pipeline — 7-Class Market Signal")
    print("=" * 70)

    # Load all raw texts
    print("\nLoading raw text data...")
    texts = load_all_raw_texts()
    print(f"\n  Total texts loaded: {len(texts)}")

    if not texts:
        print("\nNo texts found. Run setup_datasets.py and/or collection scripts first.")
        return

    # Apply weak labeling
    print("\nApplying weak labeling rules...")
    labeled = []
    for t in texts:
        signal, confidence, matched_kws = classify_text(t["text"], t.get("query", ""))
        labeled.append({
            "text": t["text"],
            "signal": signal,
            "confidence": confidence,
            "matched_keywords": "|".join(matched_kws),
            "source": t["source"],
            "timestamp": t["timestamp"],
            "area_hint": t["area_hint"],
            "business_hint": t["business_hint"],
            "query": t.get("query", ""),
            "original_label": t.get("original_label", ""),
        })

    # Statistics
    signal_counts = Counter(item["signal"] for item in labeled)
    print(f"\nSignal Distribution:")
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

    print(f"\nSaved {len(labeled)} labeled texts to {output_file}")

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

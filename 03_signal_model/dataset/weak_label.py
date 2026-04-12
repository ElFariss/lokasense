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

from common.bootstrap_utils import stable_split
from common.text_normalization import is_probably_indonesian, normalize_for_dedupe, normalize_text

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "labeled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_COMPLAINT_HINTS = {
    "makanan", "minuman", "menu", "resto", "restoran", "warung", "kedai",
    "cafe", "kopi", "bakso", "ayam", "mie", "laundry", "porsi", "rasa", "rasanya",
    "masakan", "sambal", "nasi", "kue", "barista", "kasir", "pelayan", "pelayanan",
    "warteg", "bakery", "dimsum", "rawon", "soto", "bebek", "lalapan", "seafood",
}
LOCAL_COMPLAINT_EXCLUDES = {
    "indihome", "internet", "kamera", "xiaomi", "facebook", "instagram", "twitter",
    "pesawat", "kota tua", "film indonesia", "iklan", "apbi", "beruang indonesia",
}


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
        "allow_query_context": True,
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
        "negative_keywords": [
            "mahal banget", "mahal bgt", "kecewa", "mengecewakan", "gak worth",
            "ga worth", "pelayanan lama", "pelayanan buruk", "porsi kecil",
            "kurang enak", "ga rekomen", "gak rekomen", "zonk",
        ],
        "allow_query_context": True,
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
        "allow_query_context": True,
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
        "allow_query_context": True,
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
            "terlalu mahal", "mahal doang", "ga sesuai harga",
            "gak sesuai harga", "tidak sesuai harga", "ga sebanding",
            "gak sebanding", "tidak sebanding", "kurang worth it",
            "kurang worth", "ga rekomen", "gak rekomen",
            "ga recommended", "gak recommended", "not recommended",
            "zonk", "pelit", "porsi dikit", "porsi sedikit",
            "pelayanannya lama", "pelayanannya buruk", "pelayanannya jutek",
            "ga ramah", "gak ramah", "tidak ramah", "cuek",
            "nunggu lama", "antri lama", "antre lama", "dateng lama",
            "keasinan", "kemanisan", "terlalu asin", "terlalu manis",
            "alot", "keras", "amis", "ga lagi", "gak lagi", "nggak lagi",
            "b aja sih", "ga puas", "gak puas", "kurang puas",
        ],
        "negative_keywords": [
            "murah meriah", "enak banget", "wajib coba", "recommended",
            "rekomen", "worth it", "mantap", "pelayanannya cepat",
            "pelayanannya satset", "ramah banget",
        ],
        "regex_patterns": [
            (r"\b(?:ga|gak|nggak|tidak)\s+(?:worth|rekomen|recommended)\b", 1.5),
            (r"\b(?:ga|gak|nggak|tidak)\s+sesuai\s+harga\b", 1.5),
            (r"\b(?:terlalu|kelewat)\s+mahal\b", 1.5),
            (r"\b(?:pelayanan|layanannya)\s+(?:lama|buruk|jutek|cuek)\b", 1.5),
            (r"\b(?:porsi|portion)\s+(?:kecil|dikit|sedikit)\b", 1.5),
            (r"\b(?:terlalu\s+asin|keasinan|terlalu\s+manis|kemanisan)\b", 1.5),
        ],
        "allow_query_context": False,
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
        "allow_query_context": True,
    },
}

SIGNAL_PRIORITY = {
    "COMPLAINT": 7,
    "DEMAND_UNMET": 6,
    "COMPETITION_HIGH": 5,
    "SUPPLY_SIGNAL": 4,
    "TREND": 3,
    "DEMAND_PRESENT": 2,
    "NEUTRAL": 1,
}


def contains_keyword(text: str, keyword: str) -> bool:
    pattern = re.escape(keyword.lower()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){pattern}(?!\w)", text) is not None


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
            if contains_keyword(text_lower, neg_kw):
                excluded = True
                break

        if excluded:
            continue

        # Count keyword matches
        matched = []
        context_matched = []
        for kw in rules["keywords"]:
            if contains_keyword(text_lower, kw):
                matched.append(kw)
            elif rules.get("allow_query_context", True) and context_lower and contains_keyword(context_lower, kw):
                context_matched.append(f"query:{kw}")

        regex_score = 0.0
        regex_matched = []
        for pattern, weight in rules.get("regex_patterns", []):
            if re.search(pattern, text_lower):
                regex_score += float(weight)
                regex_matched.append(f"regex:{pattern}")

        score = len(matched) + (0.75 * len(context_matched)) + regex_score
        if score > 0:
            scores[signal] = score
            matches[signal] = matched + context_matched + regex_matched

    if not scores:
        return "NEUTRAL", 0.0, []

    # Get the signal with the most keyword matches
    best_signal = max(scores, key=lambda signal: (scores[signal], SIGNAL_PRIORITY.get(signal, 0)))
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
    Returns list of dicts with text plus the metadata needed by downstream
    splitting, scoring, and spatial resolution.
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
            "platform": row.get("platform", default_source),
            "url": row.get("url", ""),
            "timestamp": row.get("timestamp", ""),
            "city": row.get("city", ""),
            "area_hint": row.get("area_hint", ""),
            "business_hint": row.get("business_hint", ""),
            "query": row.get("query", ""),
            "query_intent": row.get("query_intent", ""),
            "provenance_split": row.get("provenance_split", stable_split(text)),
            "original_label": row.get("original_label", ""),
        })

    def load_local_complaint_texts() -> None:
        corpus_specs = [
            (
                DATA_DIR / "huggingface" / "indonesian_sentiment" / "train.csv",
                "indonesian_sentiment_negative",
                "text",
                lambda row: str(row.get("label", "")) == "0",
            ),
            (
                DATA_DIR / "huggingface" / "indonesian_sentiment" / "validation.csv",
                "indonesian_sentiment_negative",
                "text",
                lambda row: str(row.get("label", "")) == "0",
            ),
            (
                DATA_DIR / "huggingface" / "indonesian_sentiment" / "test.csv",
                "indonesian_sentiment_negative",
                "text",
                lambda row: str(row.get("label", "")) == "0",
            ),
            (
                DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian" / "train.csv",
                "nusax_negative",
                "text",
                lambda row: str(row.get("label", "")).lower() == "negative",
            ),
            (
                DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian" / "valid.csv",
                "nusax_negative",
                "text",
                lambda row: str(row.get("label", "")).lower() == "negative",
            ),
            (
                DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian" / "test.csv",
                "nusax_negative",
                "text",
                lambda row: str(row.get("label", "")).lower() == "negative",
            ),
        ]

        added = 0
        for filepath, source_name, text_column, is_negative in corpus_specs:
            if not filepath.exists():
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as handle:
                    rows = csv.DictReader(handle)
                    for row in rows:
                        if not is_negative(row):
                            continue
                        text = str(row.get(text_column, "")).strip()
                        if len(text) <= 15:
                            continue
                        normalized = normalize_text(text)
                        if not is_probably_indonesian(normalized):
                            continue
                        if any(exclude_hint in normalized for exclude_hint in LOCAL_COMPLAINT_EXCLUDES):
                            continue
                        if not any(hint in normalized for hint in LOCAL_COMPLAINT_HINTS):
                            continue
                        add_text(
                            {
                                "text": normalized,
                                "source": source_name,
                                "platform": "local_corpus",
                                "url": "",
                                "timestamp": "",
                                "city": "",
                                "area_hint": "",
                                "business_hint": "",
                                "query": "",
                                "query_intent": "complaint_corpus",
                                "original_label": row.get("label", ""),
                            },
                            source_name,
                        )
                        added += 1
            except Exception:
                continue
        print(f"  Loaded local complaint augmentation: {added} texts")

    bootstrap_file = DATA_DIR / "scraped" / "signal_bootstrap.csv"
    if bootstrap_file.exists():
        with open(bootstrap_file, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                add_text(row, row.get("platform", "scraped_bootstrap"))
        scraped_count = len(texts)
        print(f"  Loaded scraped bootstrap: {scraped_count} texts")
        load_local_complaint_texts()
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

    load_local_complaint_texts()
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
            "platform": t.get("platform", ""),
            "url": t.get("url", ""),
            "timestamp": t["timestamp"],
            "city": t.get("city", ""),
            "area_hint": t["area_hint"],
            "business_hint": t["business_hint"],
            "query": t.get("query", ""),
            "query_intent": t.get("query_intent", ""),
            "provenance_split": t.get("provenance_split", stable_split(t["text"])),
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

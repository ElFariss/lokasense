#!/usr/bin/env python3
"""
Local IndoBERT self-training pseudolabel generation for market signals.

This path avoids API quotas and focuses on train-only pseudolabel expansion.
Unlike the earlier generic miner, this version actively targets harder classes
such as DEMAND_UNMET, SUPPLY_SIGNAL, COMPETITION_HIGH, and TREND by mining
candidate texts with class-specific hint patterns before asking the saved
IndoBERT checkpoint to score them.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.bootstrap_utils import text_candidates
from common.text_normalization import is_probably_indonesian, normalize_for_dedupe, normalize_text

_WEAK_LABEL_PATH = BASE_DIR / "03_signal_model" / "dataset" / "weak_label.py"
_WEAK_LABEL_SPEC = importlib.util.spec_from_file_location("weak_label", _WEAK_LABEL_PATH)
if _WEAK_LABEL_SPEC is None or _WEAK_LABEL_SPEC.loader is None:
    raise RuntimeError(f"Unable to load helper module at {_WEAK_LABEL_PATH}")
_WEAK_LABEL_MODULE = importlib.util.module_from_spec(_WEAK_LABEL_SPEC)
_WEAK_LABEL_SPEC.loader.exec_module(_WEAK_LABEL_MODULE)
SIGNAL_RULES = _WEAK_LABEL_MODULE.SIGNAL_RULES
contains_keyword = _WEAK_LABEL_MODULE.contains_keyword

LABELED_DIR = BASE_DIR / "data" / "labeled"
SOCIAL_DIR = BASE_DIR / "data" / "social_media"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "signal_base"
OUTPUT_FILE = LABELED_DIR / "model_pseudo_augmented.csv"

SIGNAL_LABELS = ["NEUTRAL", "DEMAND_UNMET", "DEMAND_PRESENT", "SUPPLY_SIGNAL", "COMPETITION_HIGH", "COMPLAINT", "TREND"]
TARGET_LABELS = ["DEMAND_UNMET", "SUPPLY_SIGNAL", "COMPETITION_HIGH", "TREND", "COMPLAINT", "DEMAND_PRESENT"]
RAW_SOURCE_LIMITS = {
    "indonesian_sentiment": 2500,
    "nusax": 500,
    "gmaps_cache": 900,
    "tiktok_scrapling": 400,
}
SOURCE_PRIORITY = {
    "tiktok_scrapling": 5,
    "gmaps_cache": 4,
    "indonesian_sentiment": 3,
    "nusax": 2,
}
LABEL_CONFIG = {
    "DEMAND_UNMET": {"candidate_quota": 260, "min_confidence": 0.86, "min_margin": 0.10, "max_keep": 120},
    "SUPPLY_SIGNAL": {"candidate_quota": 220, "min_confidence": 0.84, "min_margin": 0.08, "max_keep": 100},
    "COMPETITION_HIGH": {"candidate_quota": 220, "min_confidence": 0.84, "min_margin": 0.08, "max_keep": 100},
    "TREND": {"candidate_quota": 220, "min_confidence": 0.84, "min_margin": 0.08, "max_keep": 100},
    "COMPLAINT": {"candidate_quota": 180, "min_confidence": 0.92, "min_margin": 0.15, "max_keep": 80},
    "DEMAND_PRESENT": {"candidate_quota": 160, "min_confidence": 0.95, "min_margin": 0.22, "max_keep": 60},
}
GLOBAL_HINT_TERMS = [
    "belum ada", "ga ada", "gak ada", "tolong buka", "kapan buka", "butuh", "nyari",
    "cabang", "outlet", "gerai", "sudah ada", "udah ada", "tersedia",
    "banyak banget", "saingan", "kompetitor", "persaingan", "jenuh", "menjamur",
    "viral", "fyp", "trending", "trend", "hits", "rame", "ramai", "antri", "antre",
    "mahal", "kecewa", "pelayanan", "worth it", "enak banget", "recommended", "rekomen",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local IndoBERT pseudolabels for signal training.")
    parser.add_argument("--max-samples", type=int, default=800, help="Maximum candidate texts to score after class-targeted mining.")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite model_pseudo_augmented.csv instead of appending unique rows.")
    return parser.parse_args()


def load_existing_text_keys() -> set[str]:
    keys: set[str] = set()
    for filename in ["weak_labeled.csv", "gemini_augmented.csv", "model_pseudo_augmented.csv"]:
        path = LABELED_DIR / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text = normalize_for_dedupe(str(row.get("text", "")))
                if text:
                    keys.add(text)
    return keys


def iter_local_raw_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    corpus_specs = [
        (
            "indonesian_sentiment",
            [
                DATA_DIR / "huggingface" / "indonesian_sentiment" / "train.csv",
                DATA_DIR / "huggingface" / "indonesian_sentiment" / "validation.csv",
                DATA_DIR / "huggingface" / "indonesian_sentiment" / "test.csv",
            ],
            "text",
            "label",
        ),
        (
            "nusax",
            [
                DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian" / "train.csv",
                DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian" / "valid.csv",
                DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian" / "test.csv",
            ],
            "text",
            "label",
        ),
    ]

    for source_name, paths, text_column, label_column in corpus_specs:
        for path in paths:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            limit = RAW_SOURCE_LIMITS.get(source_name, len(df))
            if len(df) > limit:
                df = df.sample(n=limit, random_state=42)
            if text_column not in df.columns:
                continue
            for _, item in df.iterrows():
                rows.append(
                    {
                        "raw_text": str(item.get(text_column, "")),
                        "source": source_name,
                        "original_label": str(item.get(label_column, "")),
                        "city": "",
                        "area_hint": "",
                        "business_hint": "",
                        "query": "",
                        "platform": "",
                        "timestamp": "",
                        "url": "",
                    }
                )

    social_specs = [
        (SOCIAL_DIR / "tiktok_data.csv", "tiktok_scrapling", "raw_text"),
        (SOCIAL_DIR / "gmaps_reviews.csv", "gmaps_cache", "text"),
    ]
    for path, source_name, text_column in social_specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        limit = RAW_SOURCE_LIMITS.get(source_name, len(df))
        if len(df) > limit:
            df = df.sample(n=limit, random_state=42)
        if text_column not in df.columns:
            continue
        for _, item in df.iterrows():
            rows.append(
                {
                    "raw_text": str(item.get(text_column, "")),
                    "source": source_name,
                    "original_label": "",
                    "city": str(item.get("city", "")),
                    "area_hint": str(item.get("area_hint", "")),
                    "business_hint": str(item.get("business_hint", "")),
                    "query": str(item.get("query", "")),
                    "platform": str(item.get("platform", "")),
                    "timestamp": str(item.get("timestamp", "")),
                    "url": str(item.get("url", "")),
                }
            )

    return rows


def normalize_candidate(text: str) -> str:
    normalized = normalize_text(text)
    token_count = len(normalized.split())
    if len(normalized) < 12 or len(normalized) > 220:
        return ""
    if token_count < 3 or token_count > 42:
        return ""
    if not is_probably_indonesian(normalized):
        return ""
    return normalized


def score_text_for_label(text: str, label: str) -> float:
    text_lower = str(text).lower()
    rules = SIGNAL_RULES[label]

    for neg_kw in rules.get("negative_keywords", []):
        if contains_keyword(text_lower, neg_kw):
            return 0.0

    score = 0.0
    for kw in rules.get("keywords", []):
        if contains_keyword(text_lower, kw):
            score += 1.0

    for pattern, weight in rules.get("regex_patterns", []):
        if _WEAK_LABEL_MODULE.re.search(pattern, text_lower):
            score += float(weight)

    return score


def hinted_label_for_text(text: str) -> tuple[str, float]:
    best_label = ""
    best_score = 0.0
    for label in TARGET_LABELS:
        score = score_text_for_label(text, label)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score


def raw_row_is_worth_expanding(raw_text: str, source: str) -> bool:
    normalized = normalize_text(raw_text)
    if len(normalized) < 12:
        return False
    if source in {"tiktok_scrapling", "gmaps_cache"}:
        return True
    return any(term in normalized for term in GLOBAL_HINT_TERMS)


def build_targeted_candidate_pool(max_samples: int) -> list[dict[str, object]]:
    existing_keys = load_existing_text_keys()
    bucketed: dict[str, list[dict[str, object]]] = {label: [] for label in TARGET_LABELS}
    seen: set[str] = set()

    for raw_row in iter_local_raw_rows():
        if not raw_row_is_worth_expanding(str(raw_row.get("raw_text", "")), str(raw_row.get("source", ""))):
            continue
        for fragment in text_candidates(str(raw_row.get("raw_text", ""))):
            normalized = normalize_candidate(fragment)
            if not normalized:
                continue
            dedupe_key = normalize_for_dedupe(normalized)
            if not dedupe_key or dedupe_key in existing_keys or dedupe_key in seen:
                continue
            hint_label, hint_score = hinted_label_for_text(normalized)
            if not hint_label or hint_score <= 0:
                continue
            seen.add(dedupe_key)
            candidate = {
                "text": normalized,
                "source": str(raw_row.get("source", "")),
                "timestamp": str(raw_row.get("timestamp", "")),
                "city": str(raw_row.get("city", "")),
                "area_hint": str(raw_row.get("area_hint", "")),
                "business_hint": str(raw_row.get("business_hint", "")),
                "query": str(raw_row.get("query", "")),
                "platform": str(raw_row.get("platform", "")),
                "url": str(raw_row.get("url", "")),
                "original_label": str(raw_row.get("original_label", "")),
                "hint_label": hint_label,
                "hint_score": float(hint_score),
            }
            bucketed[hint_label].append(candidate)

    selected: list[dict[str, object]] = []
    selected_keys: set[str] = set()
    for label in TARGET_LABELS:
        quota = LABEL_CONFIG[label]["candidate_quota"]
        bucket = sorted(
            bucketed[label],
            key=lambda row: (
                -float(row["hint_score"]),
                -SOURCE_PRIORITY.get(str(row["source"]), 0),
                str(row["text"]),
            ),
        )
        for row in bucket:
            dedupe_key = normalize_for_dedupe(str(row["text"]))
            if not dedupe_key or dedupe_key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(dedupe_key)
            if sum(1 for item in selected if item["hint_label"] == label) >= quota:
                break

    selected = sorted(
        selected,
        key=lambda row: (
            -float(row["hint_score"]),
            -SOURCE_PRIORITY.get(str(row["source"]), 0),
            str(row["text"]),
        ),
    )
    return selected[:max_samples]


def predict_candidates(rows: list[dict[str, object]], batch_size: int) -> list[dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()

    predicted_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            enc = tokenizer(
                [str(row["text"]) for row in batch],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            top_values, top_indices = torch.topk(probs, k=2, dim=-1)
            for row, top_val, top_idx in zip(batch, top_values, top_indices):
                label = SIGNAL_LABELS[int(top_idx[0])]
                runner_up = SIGNAL_LABELS[int(top_idx[1])]
                confidence = float(top_val[0])
                runner_up_conf = float(top_val[1])
                predicted_rows.append(
                    {
                        **row,
                        "model_signal": label,
                        "model_confidence": confidence,
                        "model_margin": confidence - runner_up_conf,
                        "runner_up_signal": runner_up,
                        "runner_up_confidence": runner_up_conf,
                    }
                )
    return predicted_rows


def filter_predictions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    counts: defaultdict[str, int] = defaultdict(int)

    for row in sorted(
        rows,
        key=lambda item: (
            -float(item["model_confidence"]),
            -float(item["model_margin"]),
            -float(item["hint_score"]),
            str(item["text"]),
        ),
    ):
        hint_label = str(row["hint_label"])
        predicted_label = str(row["model_signal"])
        if predicted_label != hint_label:
            continue

        config = LABEL_CONFIG[predicted_label]
        confidence = float(row["model_confidence"])
        margin = float(row["model_margin"])
        if confidence < float(config["min_confidence"]) or margin < float(config["min_margin"]):
            continue
        if counts[predicted_label] >= int(config["max_keep"]):
            continue

        counts[predicted_label] += 1
        row["label_source"] = "model_pseudo_augmented"
        row["provenance_split"] = "train"
        kept.append(row)
    return kept


def main() -> None:
    args = parse_args()
    if not MODEL_DIR.exists():
        print(f"Model directory not found: {MODEL_DIR}")
        return

    candidate_pool = build_targeted_candidate_pool(args.max_samples)

    print("=" * 70)
    print(" IndoBERT Self-Training Pseudolabels")
    print("=" * 70)
    print(f"  Candidates to score: {len(candidate_pool)}")
    if not candidate_pool:
        print("  No net-new targeted candidate rows available.")
        return

    print(f"  Source mix: {dict(Counter(str(row.get('source', '')) for row in candidate_pool))}")
    print(f"  Hint mix: {dict(Counter(str(row.get('hint_label', '')) for row in candidate_pool))}")

    predicted_rows = predict_candidates(candidate_pool, batch_size=args.batch_size)
    kept_rows = filter_predictions(predicted_rows)

    if not kept_rows:
        print("  No rows passed the targeted confidence and margin filters.")
        return

    kept_df = pd.DataFrame(kept_rows)
    print(f"  Kept rows: {len(kept_df)}")
    print(f"  Signal mix: {kept_df['model_signal'].value_counts().to_dict()}")
    print(f"  Source mix after filtering: {kept_df['source'].value_counts().to_dict()}")

    if not args.overwrite and OUTPUT_FILE.exists():
        existing_df = pd.read_csv(OUTPUT_FILE)
        combined_df = pd.concat([existing_df, kept_df], ignore_index=True)
        combined_df["_dedupe_key"] = combined_df["text"].map(normalize_for_dedupe)
        combined_df = combined_df.drop_duplicates("_dedupe_key", keep="last").drop(columns="_dedupe_key")
    else:
        combined_df = kept_df

    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(combined_df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

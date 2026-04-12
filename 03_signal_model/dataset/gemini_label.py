#!/usr/bin/env python3
"""
Gemini-powered pseudo-labeling for 7-class market signal classification.

This script now serves two distinct roles:
1. Refinement: relabel existing weak labels from the current scraped corpus.
2. Augmentation: add fresh Gemini-labeled Indonesian market text from the
   current scraped corpus plus optional local Indonesian review datasets.

The maintained training path should consume `gemini_augmented.csv`, not the
legacy `gemini_labeled.csv` artifact from older experiments.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.bootstrap_utils import stable_hash, stable_split
from common.market_catalog import BUSINESS_HINTS, CITIES_KECAMATAN
from common.text_normalization import is_probably_indonesian, normalize_for_dedupe, normalize_text

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
LABELED_DIR = DATA_DIR / "labeled"
LABELED_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_OUTPUT_FILE = LABELED_DIR / "gemini_labeled.csv"
OUTPUT_FILE = LABELED_DIR / "gemini_augmented.csv"
CHECKPOINT_FILE = LABELED_DIR / "gemini_augmented_checkpoint.csv"
ERROR_FILE = LABELED_DIR / "gemini_augmented_errors.csv"

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite-preview")
MAX_PROMPT_TEXT_CHARS = int(os.getenv("GEMINI_MAX_TEXT_CHARS", "420"))
MODEL_LIMITS = {
    "gemini-3.1-flash-lite-preview": {"rpm": 15, "tpm": 250_000, "rpd": 500},
    "gemini-3-flash-preview": {"rpm": 5, "tpm": 250_000, "rpd": 20},
    "gemini-2.5-flash": {"rpm": 5, "tpm": 250_000, "rpd": 20},
}

SIGNAL_LABELS = [
    "DEMAND_UNMET",
    "DEMAND_PRESENT",
    "SUPPLY_SIGNAL",
    "COMPETITION_HIGH",
    "COMPLAINT",
    "TREND",
    "NEUTRAL",
]

MARKET_CONTEXT_HINTS = set(BUSINESS_HINTS) | {
    "cafe",
    "resto",
    "restoran",
    "warung",
    "kuliner",
    "tempat makan",
    "kedai",
    "bakso",
    "ayam",
    "mie",
    "kopi",
    "pelayanan",
    "harga",
    "murah",
    "mahal",
    "viral",
    "ramai",
    "rame",
    "buka",
    "cabang",
    "saingan",
    "kompetitor",
    "review",
    "worth it",
}
LOCATION_HINTS = {city.lower() for city in CITIES_KECAMATAN} | {
    district.lower()
    for districts in CITIES_KECAMATAN.values()
    for district in districts
}


SIGNAL_DEFINITIONS = """
Kamu adalah ahli analisis sentimen pasar Indonesia yang sangat tepat.
Tugas: klasifikasikan teks ulasan atau posting publik ke SATU dari 7 sinyal pasar berikut:

1. DEMAND_UNMET — Permintaan yang belum terpenuhi. Seseorang mengeluh tidak ada, belum tersedia, atau ingin tetapi tidak bisa mendapatkannya di lokasi tersebut.
2. DEMAND_PRESENT — Permintaan yang sudah ada dan terpenuhi dengan baik. Ulasan positif tentang pengalaman yang memuaskan.
3. SUPPLY_SIGNAL — Observasi faktual tentang jumlah atau ketersediaan bisnis serupa.
4. COMPETITION_HIGH — Persepsi subjektif bahwa persaingan terlalu tinggi atau jenuh.
5. COMPLAINT — Keluhan tentang kualitas, harga, atau layanan.
6. TREND — Sinyal viral, trending, atau timing.
7. NEUTRAL — Tidak mengandung sinyal pasar yang jelas.

PENTING:
- Jawab hanya dengan format JSON: {"signal": "SIGNAL_NAME", "confidence": 0.0-1.0, "reason": "alasan singkat"}
- Confidence 0.9-1.0 = sangat yakin, 0.7-0.9 = cukup yakin, <0.7 = kurang yakin
- Jika teks ambigu, berikan confidence rendah
- SUPPLY_SIGNAL vs COMPETITION_HIGH: SUPPLY = fakta (misal ada 3 toko), COMPETITION = opini (misal terlalu banyak)
"""

FEW_SHOT_EXAMPLES = [
    {"text": "Di Malang belum ada dimsum yang enak kayak di Jakarta", "signal": "DEMAND_UNMET", "confidence": 0.95, "reason": "Ekspresi kebutuhan yang belum terpenuhi di lokasi spesifik"},
    {"text": "Bakso Pak Min emang enak banget, murah meriah mantap", "signal": "DEMAND_PRESENT", "confidence": 0.92, "reason": "Ulasan sangat positif tentang tempat yang sudah ada"},
    {"text": "Udah ada 3 outlet Mixue sekarang di Lowokwaru", "signal": "SUPPLY_SIGNAL", "confidence": 0.90, "reason": "Observasi faktual tentang jumlah outlet"},
    {"text": "Banyak banget yang jualan ayam geprek di sini, dimana mana", "signal": "COMPETITION_HIGH", "confidence": 0.88, "reason": "Persepsi subjektif bahwa persaingan terlalu tinggi"},
    {"text": "Mahal banget, porsi kecil, ga worth it sih", "signal": "COMPLAINT", "confidence": 0.93, "reason": "Keluhan tentang harga dan porsi"},
    {"text": "Lagi viral di TikTok cafe baru ini, FYP mulu", "signal": "TREND", "confidence": 0.91, "reason": "Sinyal viral di media sosial"},
    {"text": "Buka dari jam 8 pagi sampai jam 10 malam", "signal": "NEUTRAL", "confidence": 0.95, "reason": "Informasi faktual tanpa sinyal pasar"},
]


def load_gemini_api_keys() -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    direct_keys = [os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_API_KEYS", "")]
    for item in direct_keys:
        for key in str(item).split(","):
            key = key.strip()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)

    for env_name, env_value in sorted(os.environ.items()):
        if not env_name.startswith("GEMINI_API_"):
            continue
        value = str(env_value).strip()
        if value and value not in seen:
            seen.add(value)
            keys.append(value)
    return keys


def init_gemini() -> dict[str, object]:
    import google.generativeai as genai

    api_keys = load_gemini_api_keys()
    if not api_keys:
        raise ValueError("No Gemini API keys found in .env file.")

    model_name = DEFAULT_GEMINI_MODEL.strip() or "gemini-3.1-flash-lite-preview"
    limits = MODEL_LIMITS.get(model_name, {})
    rpm = int(os.getenv("GEMINI_RPM", str(limits.get("rpm", 5))))
    tpm = int(os.getenv("GEMINI_TPM", str(limits.get("tpm", 250_000))))
    rpd = int(os.getenv("GEMINI_RPD", str(limits.get("rpd", 20))))
    key_count = max(1, len(api_keys))
    request_interval = float(os.getenv("GEMINI_REQUEST_INTERVAL_SEC", f"{60.0 / max(1, rpm * key_count):.3f}"))

    return {
        "genai": genai,
        "api_keys": api_keys,
        "next_index": 0,
        "model_name": model_name,
        "rpm": rpm,
        "tpm": tpm,
        "rpd": rpd,
        "request_interval_sec": request_interval,
        "daily_budget": key_count * rpd if rpd > 0 else 0,
    }


def gemini_model_for_key(pool: dict[str, object], api_key: str):
    genai = pool["genai"]
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=str(pool["model_name"]),
        generation_config={
            "temperature": 0.1,
            "top_p": 0.8,
            "max_output_tokens": 200,
            "response_mime_type": "application/json",
        },
    )


def generate_content_with_timeout(model, prompt: str, timeout_sec: int = 30):
    if hasattr(signal, "SIGALRM"):
        previous_handler = signal.getsignal(signal.SIGALRM)

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Gemini request timed out after {timeout_sec} seconds")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(max(1, int(math.ceil(timeout_sec))))
        try:
            return model.generate_content(prompt, request_options={"timeout": timeout_sec})
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    return model.generate_content(prompt, request_options={"timeout": timeout_sec})


def parse_response_payload(text_response: str) -> dict[str, object] | None:
    raw_text = str(text_response or "").strip()
    if not raw_text:
        return None

    candidates = [raw_text]

    stripped = raw_text.strip("` \n\t")
    if stripped.startswith("json"):
        stripped = stripped[4:].strip()
    if stripped and stripped != raw_text:
        candidates.append(stripped)

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw_text[start : end + 1])

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

    signal_match = re.search(
        r"(?:\"?signal\"?\s*[:=]\s*\"?([A-Z_]+)\"?)|(" + "|".join(re.escape(label) for label in SIGNAL_LABELS) + r")",
        raw_text,
        flags=re.IGNORECASE,
    )
    if not signal_match:
        return None

    signal_value = signal_match.group(1) or signal_match.group(2) or ""
    signal_value = signal_value.upper().strip()
    if signal_value not in SIGNAL_LABELS:
        return None

    confidence_match = re.search(r"\"?confidence\"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", raw_text, flags=re.IGNORECASE)
    reason_match = re.search(r"\"?reason\"?\s*[:=]\s*\"([^\"]+)\"", raw_text, flags=re.IGNORECASE)
    return {
        "signal": signal_value,
        "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
        "reason": reason_match.group(1).strip() if reason_match else "parsed from semi-structured response",
    }


def classify_with_gemini(pool: dict[str, object], text: str) -> dict[str, object]:
    prompt_text = str(text).strip()
    if len(prompt_text) > MAX_PROMPT_TEXT_CHARS:
        prompt_text = prompt_text[:MAX_PROMPT_TEXT_CHARS].rsplit(" ", 1)[0].strip()

    examples_str = "\n".join(
        [
            f'Teks: "{example["text"]}"\nJawab: {{"signal": "{example["signal"]}", "confidence": {example["confidence"]}, "reason": "{example["reason"]}"}}'
            for example in FEW_SHOT_EXAMPLES
        ]
    )
    prompt = f"""{SIGNAL_DEFINITIONS}

Contoh:
{examples_str}

Sekarang klasifikasikan teks berikut:
Teks: "{prompt_text}"
Jawab:"""

    api_keys = list(pool["api_keys"])
    start_index = int(pool["next_index"])
    last_error = ""
    quota_error_count = 0

    for offset in range(len(api_keys)):
        key_index = (start_index + offset) % len(api_keys)
        api_key = api_keys[key_index]
        response = None
        try:
            model = gemini_model_for_key(pool, api_key)
            response = generate_content_with_timeout(model, prompt, timeout_sec=30)
            raw_response_text = response.text if response is not None and hasattr(response, "text") else ""
            result = parse_response_payload(raw_response_text)
            if not isinstance(result, dict):
                raise json.JSONDecodeError("unable to parse Gemini response", raw_response_text, 0)
            signal = result.get("signal", "NEUTRAL")
            if signal not in SIGNAL_LABELS:
                signal = "NEUTRAL"
            pool["next_index"] = (key_index + 1) % len(api_keys)
            return {
                "ok": True,
                "signal": signal,
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", ""),
                "error": "",
                "quota_exhausted": False,
            }
        except json.JSONDecodeError:
            text_response = response.text if response is not None and hasattr(response, "text") else ""
            for signal in SIGNAL_LABELS:
                if signal in text_response:
                    pool["next_index"] = (key_index + 1) % len(api_keys)
                    return {
                        "ok": True,
                        "signal": signal,
                        "confidence": 0.5,
                        "reason": "parsed from non-JSON",
                        "error": "",
                        "quota_exhausted": False,
                    }
            last_error = "failed to parse"
        except Exception as exc:
            last_error = str(exc)
            lowered = last_error.lower()
            if "429" in lowered or "quota" in lowered or "rate limit" in lowered:
                quota_error_count += 1
            continue

    print(f"  Gemini error across {len(api_keys)} keys: {last_error}")
    return {
        "ok": False,
        "signal": "",
        "confidence": 0.0,
        "reason": "",
        "error": last_error,
        "quota_exhausted": quota_error_count >= len(api_keys) and len(api_keys) > 0,
    }


def relevance_score(text: str) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 0

    score = 0
    for hint in MARKET_CONTEXT_HINTS:
        if hint in normalized:
            score += 2
    for hint in LOCATION_HINTS:
        if hint in normalized:
            score += 1
    for keyword in ["enak", "murah", "mahal", "pelayanan", "viral", "rame", "ramai", "saingan", "cabang", "review"]:
        if keyword in normalized:
            score += 1
    return score


def normalized_row(
    text: str,
    *,
    source: str,
    timestamp: str = "",
    city: str = "",
    area_hint: str = "",
    business_hint: str = "",
    query: str = "",
    platform: str = "",
    url: str = "",
    weak_signal: str = "",
    weak_confidence: str = "",
    original_label: str = "",
    label_source: str = "",
) -> dict[str, object] | None:
    normalized = normalize_text(text)
    if len(normalized) < 12 or not is_probably_indonesian(normalized):
        return None
    score = relevance_score(normalized)
    return {
        "text": normalized,
        "source": source,
        "timestamp": timestamp or "",
        "city": city or "",
        "area_hint": area_hint or "",
        "business_hint": business_hint or "",
        "query": normalize_text(query) if query else "",
        "platform": platform or "",
        "url": url or "",
        "weak_signal": weak_signal or "",
        "weak_confidence": weak_confidence or "",
        "original_label": original_label or "",
        "label_source": label_source or source,
        "provenance_split": stable_split(normalized),
        "relevance_score": score,
    }


def add_unique_row(rows: list[dict[str, object]], seen: set[str], row: dict[str, object] | None) -> None:
    if not row:
        return
    dedupe_key = normalize_for_dedupe(str(row["text"]))
    if not dedupe_key:
        return
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    rows.append(row)


def load_existing_weak_text_keys() -> set[str]:
    weak_file = LABELED_DIR / "weak_labeled.csv"
    if not weak_file.exists():
        return set()

    keys: set[str] = set()
    with open(weak_file, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            dedupe_key = normalize_for_dedupe(str(raw_row.get("text", "")))
            if dedupe_key:
                keys.add(dedupe_key)
    return keys


def load_weak_rows(mode: str, seen: set[str]) -> list[dict[str, object]]:
    weak_file = LABELED_DIR / "weak_labeled.csv"
    if not weak_file.exists():
        return []

    rows: list[dict[str, object]] = []
    with open(weak_file, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            confidence = float(raw_row.get("confidence", 0) or 0)
            include = False
            if mode == "low_confidence":
                include = confidence < 0.8 or raw_row.get("signal", "") == "NEUTRAL"
            elif mode == "refine":
                include = True
            elif mode == "label_all":
                include = True
            elif mode == "augment":
                include = confidence < 0.95 or raw_row.get("signal", "") in {"NEUTRAL", "COMPLAINT", "COMPETITION_HIGH"}
            if not include:
                continue
            add_unique_row(
                rows,
                seen,
                normalized_row(
                    raw_row.get("text", ""),
                    source=raw_row.get("source", "weak_labeled"),
                    timestamp=raw_row.get("timestamp", ""),
                    city=raw_row.get("city", ""),
                    area_hint=raw_row.get("area_hint", ""),
                    business_hint=raw_row.get("business_hint", ""),
                    query=raw_row.get("query", ""),
                    platform=raw_row.get("platform", ""),
                    url=raw_row.get("url", ""),
                    weak_signal=raw_row.get("signal", ""),
                    weak_confidence=raw_row.get("confidence", ""),
                    original_label=raw_row.get("original_label", ""),
                    label_source="weak_labeled",
                ),
            )
    return rows


def load_signal_bootstrap_rows(seen: set[str]) -> list[dict[str, object]]:
    bootstrap_file = DATA_DIR / "scraped" / "signal_bootstrap.csv"
    if not bootstrap_file.exists():
        return []

    rows: list[dict[str, object]] = []
    with open(bootstrap_file, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            add_unique_row(
                rows,
                seen,
                normalized_row(
                    raw_row.get("text", ""),
                    source=raw_row.get("source", "signal_bootstrap"),
                    timestamp=raw_row.get("timestamp", ""),
                    city=raw_row.get("city", ""),
                    area_hint=raw_row.get("area_hint", ""),
                    business_hint=raw_row.get("business_hint", ""),
                    query=raw_row.get("query", ""),
                    platform=raw_row.get("platform", ""),
                    url=raw_row.get("url", ""),
                    label_source="signal_bootstrap",
                ),
            )
    return rows


def load_local_corpus_rows(seen: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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

    for corpus_name, paths, text_column, label_column in corpus_specs:
        for path in paths:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if text_column not in df.columns:
                continue
            for _, item in df.iterrows():
                row = normalized_row(
                    str(item.get(text_column, "")),
                    source=corpus_name,
                    original_label=str(item.get(label_column, "")),
                    label_source=f"{corpus_name}_candidate",
                )
                if not row or int(row["relevance_score"]) < 2:
                    continue
                add_unique_row(rows, seen, row)
    return rows


def build_candidate_pool(mode: str, max_samples: int) -> list[dict[str, object]]:
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    weak_text_keys = load_existing_weak_text_keys() if mode == "augment" else set()

    if mode == "augment":
        bootstrap_rows = load_signal_bootstrap_rows(seen)
        corpus_rows = load_local_corpus_rows(seen)
        rows.extend(corpus_rows)
        rows.extend(bootstrap_rows)
        rows = [
            row
            for row in rows
            if normalize_for_dedupe(str(row.get("text", ""))) not in weak_text_keys
        ]
    else:
        weak_rows = load_weak_rows(mode, seen)
        rows.extend(weak_rows)

    # For augmentation we prioritize net-new local corpora first, then any
    # remaining public social rows that are not already in weak_labeled.csv.
    source_priority = {
        "indonesian_sentiment": 4,
        "nusax": 3,
        "signal_bootstrap": 2,
        "tiktok_scrapling": 2,
        "instagram_scrapling": 2,
        "x_scrapling": 2,
        "google_maps": 2,
        "weak_labeled": 1,
    }

    rows.sort(
        key=lambda row: (
            -source_priority.get(str(row.get("source", "")), 0),
            -int(row.get("relevance_score", 0)),
            stable_hash(str(row["text"])),
        )
    )

    if max_samples and len(rows) > max_samples:
        rows = rows[:max_samples]
    return rows


def classify_batch(
    pool: dict[str, object],
    texts: list[dict[str, object]],
    batch_size: int = 5,
    existing_results: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], bool]:
    results: list[dict[str, object]] = list(existing_results or [])
    errors: list[dict[str, object]] = []
    total = len(texts)
    stopped_for_quota = False
    request_interval = float(pool.get("request_interval_sec", 0.5))
    processed_before = len(results)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        for text_item in batch:
            result = classify_with_gemini(pool, str(text_item["text"]))
            if result["ok"]:
                results.append(
                    {
                        **text_item,
                        "gemini_signal": result["signal"],
                        "gemini_confidence": result["confidence"],
                        "gemini_reason": result["reason"],
                    }
                )
            else:
                errors.append(
                    {
                        **text_item,
                        "gemini_error": result["error"][:500],
                        "quota_exhausted": result["quota_exhausted"],
                    }
                )
                if result["quota_exhausted"]:
                    stopped_for_quota = True
                    print("  All configured Gemini keys are quota-exhausted for the selected model. Stopping this run without writing failed labels into the training set.")
                    break
            time.sleep(request_interval)

        if stopped_for_quota:
            break

        done = processed_before + min(start + batch_size, total)
        grand_total = processed_before + total
        print(f"  [{done}/{grand_total}] Processed {done} texts ({done / grand_total * 100:.0f}%)")
        if done % 100 == 0:
            save_rows(CHECKPOINT_FILE, results)
        time.sleep(1.0)

    return results, errors, stopped_for_quota


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_saved_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini pseudo-labeling for market signals.")
    parser.add_argument(
        "--mode",
        choices=["refine", "label_all", "low_confidence", "augment"],
        default="augment",
        help="augment builds a fresh Gemini-labeled corpus for the maintained notebook workflow.",
    )
    parser.add_argument("--max-samples", type=int, default=1200, help="Maximum texts to label in this run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for any sampling operations.")
    parser.add_argument("--resume", action="store_true", help="Resume from gemini_augmented_checkpoint.csv if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print("=" * 70)
    print(" Gemini Pseudo-Labeling — 7-Class Market Signal")
    print(f" Mode: {args.mode} | Max samples: {args.max_samples}")
    print("=" * 70)

    texts_to_label = build_candidate_pool(args.mode, args.max_samples)
    print(f"\n  Texts to label: {len(texts_to_label)}")
    if not texts_to_label:
        print("  Nothing to label.")
        return

    source_counts = Counter(str(item.get("source", "")) for item in texts_to_label)
    print(f"  Source mix: {dict(source_counts)}")

    print(f"\nInitializing Gemini model: {DEFAULT_GEMINI_MODEL}")
    try:
        pool = init_gemini()
    except ValueError as exc:
        print(f"\n{exc} Skipping Gemini augmentation.")
        return
    print(f"  Loaded {len(pool['api_keys'])} Gemini key(s)")
    if int(pool["daily_budget"]) > 0:
        print(
            "  Budget summary: "
            f"{pool['rpm']} RPM/key, {pool['tpm']} TPM/key, {pool['rpd']} RPD/key, "
            f"estimated daily request budget {pool['daily_budget']}"
        )
        if args.max_samples > int(pool["daily_budget"]):
            print(f"  Capping max samples from {args.max_samples} to daily budget {pool['daily_budget']}.")
            args.max_samples = int(pool["daily_budget"])
            texts_to_label = texts_to_label[: args.max_samples]

    existing_results: list[dict[str, object]] = []
    if args.resume and CHECKPOINT_FILE.exists():
        existing_results = load_saved_rows(CHECKPOINT_FILE)
        processed_texts = {str(item.get("text", "")).strip() for item in existing_results}
        texts_to_label = [item for item in texts_to_label if str(item.get("text", "")).strip() not in processed_texts]
        print(f"  Resuming from checkpoint with {len(existing_results)} completed rows.")
        print(f"  Remaining texts to label: {len(texts_to_label)}")

    print("\nRunning pseudo-labeling...")
    results, errors, stopped_for_quota = classify_batch(pool, texts_to_label, existing_results=existing_results)
    if errors:
        save_rows(ERROR_FILE, errors)

    if not results:
        for path in [OUTPUT_FILE, CHECKPOINT_FILE]:
            if path.exists():
                path.unlink()
        print("\nNo usable Gemini labels were produced in this run.")
        if errors:
            print(f"Saved {len(errors)} error rows to {ERROR_FILE}")
        if stopped_for_quota:
            print("The run ended early because all Gemini keys were out of quota for the selected model.")
        return

    save_rows(OUTPUT_FILE, results)

    signal_counts = Counter(str(item["gemini_signal"]) for item in results)
    print("\nGemini Signal Distribution:")
    for signal, count in signal_counts.most_common():
        pct = count / len(results) * 100
        print(f"  {signal:20s}  {count:>5}  ({pct:5.1f}%)")

    high_conf = sum(1 for item in results if float(item["gemini_confidence"]) >= 0.8)
    print(f"\n  High confidence (>=0.8): {high_conf} ({high_conf / len(results) * 100:.1f}%)")
    print(f"\nSaved {len(results)} Gemini-labeled texts to {OUTPUT_FILE}")
    if errors:
        print(f"Saved {len(errors)} Gemini error rows to {ERROR_FILE}")
    if args.mode in {"refine", "label_all", "low_confidence"}:
        save_rows(LEGACY_OUTPUT_FILE, results)
        print(f"Saved compatibility copy to {LEGACY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()

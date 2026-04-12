from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.bootstrap_utils import stable_hash
from common.text_normalization import normalize_for_dedupe, normalize_text
try:
    from .query_parser import BUSINESS_ALIAS_MAP, IntentResult
except ImportError:
    from query_parser import BUSINESS_ALIAS_MAP, IntentResult

BASE_DIR = Path(__file__).resolve().parent.parent
AIRGAP_DIR = BASE_DIR / "data" / "airgap"
AIRGAP_CORPUS = AIRGAP_DIR / "airgap_corpus.csv"
AIRGAP_MANIFEST = AIRGAP_DIR / "airgap_manifest.json"

SOURCE_FILES = [
    BASE_DIR / "data" / "labeled" / "weak_labeled.csv",
    BASE_DIR / "data" / "labeled" / "model_pseudo_augmented.csv",
    BASE_DIR / "data" / "labeled" / "gemini_augmented.csv",
    BASE_DIR / "data" / "scraped" / "signal_bootstrap.csv",
    BASE_DIR / "data" / "social_media" / "tiktok_data.csv",
]

SOURCE_GLOBS = [
    BASE_DIR / "outputs" / "live_runs",
]

STANDARD_COLUMNS = [
    "text",
    "source",
    "platform",
    "url",
    "timestamp",
    "city",
    "area_hint",
    "business_hint",
    "query",
    "query_intent",
    "label_source",
    "corpus_source",
]


@dataclass(slots=True)
class AirgapRetrievalResult:
    frame: pd.DataFrame
    rows_fetched: int
    rows_used: int
    corpus_path: Path
    manifest_path: Path


def _normalize_business(text: str) -> str:
    normalized = normalize_text(text)
    return BUSINESS_ALIAS_MAP.get(normalized, normalized)


def _standardize_frame(df: pd.DataFrame, corpus_source: str) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    if "signal" in df.columns and "final_signal" not in df.columns:
        rename_map["signal"] = "existing_signal"
    if "model_signal" in df.columns and "existing_signal" not in rename_map.values():
        rename_map["model_signal"] = "existing_signal"
    if "gemini_signal" in df.columns and "existing_signal" not in rename_map.values():
        rename_map["gemini_signal"] = "existing_signal"
    if rename_map:
        df = df.rename(columns=rename_map)

    for column in STANDARD_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df["city"] = df["city"].fillna("").astype(str).str.strip()
    df["area_hint"] = df["area_hint"].fillna("").astype(str).str.strip()
    df["business_hint"] = df["business_hint"].fillna("").astype(str).map(_normalize_business)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["corpus_source"] = corpus_source
    return df[STANDARD_COLUMNS + [column for column in ["existing_signal"] if column in df.columns]]


def _iter_live_run_files() -> list[Path]:
    live_files: list[Path] = []
    for root in SOURCE_GLOBS:
        if not root.exists():
            continue
        live_files.extend(sorted(root.glob("*/live_social_rows.csv")))
        live_files.extend(sorted(root.glob("*/labeled_live_rows.csv")))
    return live_files


def build_airgap_corpus(*, refresh: bool = False) -> tuple[pd.DataFrame, Path, Path]:
    AIRGAP_DIR.mkdir(parents=True, exist_ok=True)
    if AIRGAP_CORPUS.exists() and not refresh:
        return pd.read_csv(AIRGAP_CORPUS), AIRGAP_CORPUS, AIRGAP_MANIFEST

    frames: list[pd.DataFrame] = []
    manifest_sources: list[dict[str, object]] = []

    for path in SOURCE_FILES + _iter_live_run_files():
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "text" not in df.columns:
            continue
        standardized = _standardize_frame(df, str(path.relative_to(BASE_DIR)))
        frames.append(standardized)
        manifest_sources.append({"path": str(path.relative_to(BASE_DIR)), "rows": int(len(standardized))})

    if not frames:
        empty = pd.DataFrame(columns=STANDARD_COLUMNS)
        empty.to_csv(AIRGAP_CORPUS, index=False)
        with open(AIRGAP_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump({"sources": [], "rows": 0}, handle, indent=2, ensure_ascii=False)
        return empty, AIRGAP_CORPUS, AIRGAP_MANIFEST

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[combined["text"].astype(str).str.len() > 8].copy()
    combined["dedupe_key"] = combined["text"].astype(str).map(normalize_for_dedupe)
    combined = combined[combined["dedupe_key"].astype(str) != ""].copy()
    combined = combined.drop_duplicates(subset=["city", "area_hint", "dedupe_key"], keep="first")
    combined = combined.drop(columns=["dedupe_key"])
    combined.to_csv(AIRGAP_CORPUS, index=False)

    with open(AIRGAP_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "rows": int(len(combined)),
                "sources": manifest_sources,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return combined, AIRGAP_CORPUS, AIRGAP_MANIFEST


def _business_relevance(row: pd.Series, intent: IntentResult) -> int:
    score = 0
    target = _normalize_business(intent.business_type)
    row_business = _normalize_business(str(row.get("business_hint", "")))
    if row_business == target:
        score += 5

    text_blob = " ".join(
        [
            str(row.get("text", "")),
            str(row.get("query", "")),
            str(row.get("business_hint", "")),
        ]
    )
    normalized_blob = normalize_text(text_blob)
    target_tokens = [token for token in normalize_text(target).split() if len(token) > 2]
    if any(token in normalized_blob for token in target_tokens):
        score += 3

    alias_hits = 0
    for alias, canonical in BUSINESS_ALIAS_MAP.items():
        if canonical != target:
            continue
        if alias and alias in normalized_blob:
            alias_hits += 1
    score += min(alias_hits, 2)
    return score


def _row_relevance(row: pd.Series, intent: IntentResult) -> int:
    score = 0
    if str(row.get("city", "")).strip() == intent.city:
        score += 6
    else:
        return -100

    area_hint = str(row.get("area_hint", "")).strip()
    if area_hint and area_hint in intent.kecamatan_scope:
        score += 4

    score += _business_relevance(row, intent)

    normalized_text = normalize_text(str(row.get("text", "")))
    for token in normalize_text(intent.raw_query).split():
        if len(token) <= 3:
            continue
        if token in normalized_text:
            score += 1
    if str(row.get("corpus_source", "")).startswith("outputs/live_runs/"):
        score += 2
    return score


def collect_airgap_data(
    intent: IntentResult,
    *,
    min_rows: int = 20,
    top_k: int = 180,
    refresh_corpus: bool = False,
) -> AirgapRetrievalResult:
    corpus_df, corpus_path, manifest_path = build_airgap_corpus(refresh=refresh_corpus)
    if corpus_df.empty:
        raise ValueError("airgap_corpus_missing_or_empty")

    working = corpus_df.copy()
    working["relevance"] = working.apply(lambda row: _row_relevance(row, intent), axis=1)
    working = working[working["relevance"] > 0].copy()
    if working.empty:
        raise ValueError(f"insufficient_airgap_data: no local rows for {intent.city} / {intent.business_type}")

    working = working.sort_values(["relevance", "timestamp"], ascending=[False, False]).head(top_k)

    area_counts = working["area_hint"].fillna("").replace("", pd.NA).dropna().value_counts().to_dict()
    if len(intent.kecamatan_scope) == 1:
        required_rows = max(10, min_rows // 3)
    else:
        required_rows = min_rows

    if len(working) < required_rows:
        raise ValueError(
            f"insufficient_airgap_data: usable={len(working)}, required={required_rows}, city={intent.city}, business={intent.business_type}"
        )

    return AirgapRetrievalResult(
        frame=working.drop(columns=["relevance"]),
        rows_fetched=int(len(corpus_df)),
        rows_used=int(len(working)),
        corpus_path=corpus_path,
        manifest_path=manifest_path,
    )

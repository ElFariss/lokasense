from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_COLLECTION_DIR = BASE_DIR / "01_data_collection"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))

from common.bootstrap_utils import stable_hash, write_csv_rows
from common.bootstrap_utils import text_candidates
from common.text_normalization import is_probably_indonesian, normalize_for_dedupe, normalize_text
try:
    from .query_parser import IntentResult
except ImportError:
    from query_parser import IntentResult
from social_bootstrap import (
    DEFAULT_PLATFORMS,
    discover_tiktok_direct,
    discover_urls_via_search,
    extract_instagram_record,
    extract_instagram_record_static,
    extract_tiktok_record,
    extract_tiktok_record_static,
    extract_x_record,
    extract_x_record_static,
    import_scrapling,
    is_record_relevant,
    load_platform_cookies,
)

LIVE_TEMPLATE_SPECS = [
    {"template": "{business} {district} {city}", "query_intent": "discovery"},
    {"template": "review {business} {district} {city}", "query_intent": "review"},
    {"template": "{business} lagi hits di {district} {city}", "query_intent": "trend"},
    {"template": "belum ada {business} di {district} {city}", "query_intent": "demand"},
    {"template": "{business} mahal banget {district} {city}", "query_intent": "complaint"},
    {"template": "{business} saingan banyak di {district} {city}", "query_intent": "competition"},
]


@dataclass(slots=True)
class LiveRetrievalResult:
    frame: pd.DataFrame
    rows_fetched: int
    rows_used: int
    run_dir: Path
    manifest_path: Path
    combined_path: Path
    errors: list[str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_live_query_seeds(intent: IntentResult, max_queries: int = 12) -> list[dict[str, str]]:
    scope = list(intent.kecamatan_scope or [])
    if len(scope) > 1:
        prioritized_scope = scope[:3]
        ordered_templates = [spec for spec in LIVE_TEMPLATE_SPECS if spec["query_intent"] in {"discovery", "review"}]
    else:
        prioritized_scope = scope
        ordered_templates = LIVE_TEMPLATE_SPECS

    seeds: list[dict[str, str]] = [
        {
            "query": intent.raw_query,
            "city": intent.city,
            "area_hint": prioritized_scope[0] if len(prioritized_scope) == 1 else "",
            "business_hint": intent.business_type,
            "query_intent": "user_query",
        }
    ]
    for spec in ordered_templates:
        for district in prioritized_scope:
            if len(seeds) >= max_queries:
                return seeds[:max_queries]
            seeds.append(
                {
                    "query": spec["template"].format(
                        business=intent.business_type,
                        district=district,
                        city=intent.city,
                    ).strip(),
                    "city": intent.city,
                    "area_hint": district,
                    "business_hint": intent.business_type,
                    "query_intent": str(spec["query_intent"]),
                }
            )
    return seeds[:max_queries]


def _expand_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    seen_text_hashes: set[str] = set()
    for record in records:
        raw_value = str(record.get("raw_text") or record.get("text") or "")
        raw_candidates = text_candidates(raw_value)
        extra_candidates = []
        for chunk in re.split(r"[\n\r]+|(?<=[.!?])\s+|,\s+", raw_value):
            chunk = normalize_text(chunk)
            if 8 <= len(chunk) <= 180:
                extra_candidates.append(chunk)
        if extra_candidates:
            raw_candidates.extend(extra_candidates)
        if not raw_candidates:
            fallback = normalize_text(str(record.get("text", "")))
            raw_candidates = [fallback] if fallback else []
        for candidate_index, candidate in enumerate(raw_candidates):
            if not candidate or not is_probably_indonesian(candidate, strict=True):
                continue
            text_hash = stable_hash(normalize_for_dedupe(candidate))
            if text_hash in seen_text_hashes:
                continue
            seen_text_hashes.add(text_hash)
            expanded = dict(record)
            expanded["text"] = candidate
            expanded["candidate_index"] = str(candidate_index)
            content_id = str(expanded.get("content_id", "")).strip()
            expanded["content_id"] = f"{content_id}:{candidate_index}" if content_id else str(candidate_index)
            kept.append(expanded)
    return kept


def _dedupe_expanded_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    seen_text_hashes: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    for record in records:
        text = str(record.get("text", "")).strip()
        if not text or not is_probably_indonesian(text, strict=True):
            continue
        text_hash = stable_hash(normalize_for_dedupe(text))
        dedupe_key = (
            str(record.get("url", "")),
            str(record.get("content_id", "")),
        )
        if text_hash in seen_text_hashes or dedupe_key in seen_keys:
            continue
        seen_text_hashes.add(text_hash)
        seen_keys.add(dedupe_key)
        kept.append(record)
    return kept


def _dynamic_extractor(platform: str):
    if platform == "tiktok":
        return extract_tiktok_record
    if platform == "instagram":
        return extract_instagram_record
    return extract_x_record


def _static_extractor(platform: str):
    if platform == "tiktok":
        return extract_tiktok_record_static
    if platform == "instagram":
        return extract_instagram_record_static
    return extract_x_record_static


def _collect_platform_live(
    platform: str,
    seeds: list[dict[str, str]],
    run_dir: Path,
    deadline: float,
    *,
    headless: bool,
    max_per_query: int,
    max_saved_rows: int,
    request_timeout_ms: int,
    search_engine_only: bool,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    DynamicSession, FetcherSession = import_scrapling()
    cookies = load_platform_cookies(platform)
    scrape_mode = "cookie_session" if cookies else "public_first"
    output_file = run_dir / f"{platform}_live.csv"

    platform_report: dict[str, object] = {
        "queries": 0,
        "discovered_urls": 0,
        "rows_saved": 0,
        "duplicate_urls": 0,
        "duplicate_texts": 0,
        "blocked_reasons": [],
    }

    rows: list[dict[str, str]] = []
    seen_url_hashes: set[str] = set()
    seen_text_hashes: set[str] = set()

    dynamic_context = (
        nullcontext(None)
        if search_engine_only
        else DynamicSession(
            headless=headless,
            disable_resources=False,
            network_idle=False,
            real_chrome=False,
            cookies=cookies or None,
        )
    )

    with dynamic_context as dynamic_session, FetcherSession(impersonate="chrome") as fetch_session:
        for seed in seeds:
            if time.monotonic() >= deadline or (max_saved_rows and len(rows) >= max_saved_rows):
                break

            platform_report["queries"] = int(platform_report["queries"]) + 1
            urls: list[str] = []
            effective_mode = f"{scrape_mode}+search_engine"
            query_limit = max_per_query if seed.get("area_hint") else min(2, max_per_query)

            if platform == "tiktok":
                if not search_engine_only:
                    try:
                        urls = discover_tiktok_direct(
                            dynamic_session,
                            seed["query"],
                            query_limit,
                            timeout_ms=request_timeout_ms,
                        )
                        effective_mode = f"{scrape_mode}+direct_search"
                    except Exception as exc:
                        platform_report["blocked_reasons"].append({"query": seed["query"], "reason": f"direct_search_error: {exc}"})
                if len(urls) < min(2, query_limit):
                    extra_urls = discover_urls_via_search(fetch_session, platform, seed["query"], query_limit)
                    urls.extend(url for url in extra_urls if url not in urls)
                    effective_mode = f"{scrape_mode}+search_engine_fallback"
            else:
                urls = discover_urls_via_search(fetch_session, platform, seed["query"], query_limit)

            if not urls:
                platform_report["blocked_reasons"].append({"query": seed["query"], "reason": "no_public_urls_found"})
                continue

            for url in urls[:query_limit]:
                if time.monotonic() >= deadline or (max_saved_rows and len(rows) >= max_saved_rows):
                    break
                platform_report["discovered_urls"] = int(platform_report["discovered_urls"]) + 1
                url_hash = stable_hash(url)
                if url_hash in seen_url_hashes:
                    platform_report["duplicate_urls"] = int(platform_report["duplicate_urls"]) + 1
                    continue
                seen_url_hashes.add(url_hash)

                try:
                    if search_engine_only:
                        extractor = _static_extractor(platform)
                        record, blocked_reason = extractor(fetch_session, url, seed, effective_mode)
                    else:
                        extractor = _dynamic_extractor(platform)
                        record, blocked_reason = extractor(dynamic_session, url, seed, effective_mode, timeout_ms=request_timeout_ms)
                except Exception as exc:
                    platform_report["blocked_reasons"].append({"url": url, "reason": f"extract_error: {exc}"})
                    continue

                if blocked_reason:
                    platform_report["blocked_reasons"].append({"url": url, "reason": blocked_reason})
                if not record or not is_record_relevant(record, seed):
                    continue

                text_hash = stable_hash(normalize_for_dedupe(str(record.get("text", ""))))
                if text_hash in seen_text_hashes:
                    platform_report["duplicate_texts"] = int(platform_report["duplicate_texts"]) + 1
                    continue
                seen_text_hashes.add(text_hash)
                rows.append(record)

    rows = _expand_records(rows)
    platform_report["rows_saved"] = len(rows)
    if rows:
        write_csv_rows(output_file, rows, list(rows[0].keys()))
    return rows, platform_report


def collect_live_data(
    intent: IntentResult,
    output_dir: Path,
    *,
    timeout_sec: int = 90,
    platforms: list[str] | None = None,
    min_live_rows: int = 30,
    headless: bool = True,
    max_queries: int = 6,
    max_per_query: int = 3,
    max_saved_rows_per_platform: int = 40,
    request_timeout_ms: int = 10000,
    search_engine_only: bool = False,
) -> LiveRetrievalResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    platforms = list(platforms or DEFAULT_PLATFORMS)
    seeds = build_live_query_seeds(intent, max_queries=max_queries)
    deadline = time.monotonic() + max(5, int(timeout_sec))

    manifest: dict[str, object] = {
        "created_at": _now_iso(),
        "raw_query": intent.raw_query,
        "city": intent.city,
        "business_type": intent.business_type,
        "kecamatan_scope": intent.kecamatan_scope,
        "platforms": {},
        "errors": [],
        "query_seeds": seeds,
    }

    all_rows: list[dict[str, str]] = []
    executor = ThreadPoolExecutor(max_workers=len(platforms))
    future_map = {
        executor.submit(
            _collect_platform_live,
            platform,
            seeds,
            output_dir,
            deadline,
            headless=headless,
            max_per_query=max_per_query,
            max_saved_rows=max_saved_rows_per_platform,
            request_timeout_ms=request_timeout_ms,
            search_engine_only=search_engine_only,
        ): platform
        for platform in platforms
    }
    timed_out_platforms: set[str] = set()
    remaining = max(1.0, deadline - time.monotonic())
    done, not_done = wait(set(future_map), timeout=remaining)
    for future in not_done:
        timed_out_platforms.add(future_map[future])

    executor.shutdown(wait=True, cancel_futures=False)

    for future, platform in future_map.items():
        try:
            rows, report = future.result()
            if platform in timed_out_platforms:
                manifest["errors"].append(f"{platform}: timeout after {timeout_sec}s")
                blocked = list(report.get("blocked_reasons", []))
                blocked.append(f"timeout_after_{timeout_sec}s")
                report["blocked_reasons"] = blocked
            manifest["platforms"][platform] = report
            all_rows.extend(rows)
        except Exception as exc:
            manifest["errors"].append(f"{platform}: {exc}")
            manifest["platforms"][platform] = {
                "queries": 0,
                "discovered_urls": 0,
                "rows_saved": 0,
                "blocked_reasons": [str(exc)],
            }

    for platform in platforms:
        partial_path = output_dir / f"{platform}_live.csv"
        if not partial_path.exists():
            continue
        try:
            partial_rows = pd.read_csv(partial_path).fillna("").to_dict("records")
        except Exception:
            continue
        if not partial_rows:
            continue
        all_rows.extend(partial_rows)
        platform_report = manifest["platforms"].setdefault(
            platform,
            {
                "queries": 0,
                "discovered_urls": 0,
                "rows_saved": 0,
                "duplicate_urls": 0,
                "duplicate_texts": 0,
                "blocked_reasons": [],
            },
        )
        platform_report["rows_saved"] = max(int(platform_report.get("rows_saved", 0)), len(partial_rows))

    combined_rows = _dedupe_expanded_records(all_rows)
    combined_df = pd.DataFrame(combined_rows)
    combined_path = output_dir / "live_social_rows.csv"
    manifest_path = output_dir / "live_manifest.json"

    if not combined_df.empty:
        combined_df.to_csv(combined_path, index=False)

    rows_fetched = len(all_rows)
    rows_used = len(combined_df)
    manifest["rows_fetched"] = rows_fetched
    manifest["rows_used"] = rows_used
    manifest["completed_at"] = _now_iso()

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    if rows_used < min_live_rows:
        raise ValueError(
            f"insufficient_live_data: fetched={rows_fetched}, usable={rows_used}, minimum_required={min_live_rows}, manifest={manifest_path}"
        )

    return LiveRetrievalResult(
        frame=combined_df,
        rows_fetched=rows_fetched,
        rows_used=rows_used,
        run_dir=output_dir,
        manifest_path=manifest_path,
        combined_path=combined_path,
        errors=[str(item) for item in manifest.get("errors", [])],
    )

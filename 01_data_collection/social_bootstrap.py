#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote_plus, urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.bootstrap_utils import (
    SCRAPED_DIR,
    SOCIAL_DIR,
    build_ner_bootstrap_rows,
    build_signal_bootstrap_rows,
    canonicalize_url,
    dump_json,
    load_cookie_file,
    load_csv_rows,
    stable_hash,
    write_csv_rows,
)
from common.market_catalog import iter_query_seeds
from common.text_normalization import extract_hashtags, normalize_for_dedupe, normalize_text, strip_emoji

if TYPE_CHECKING:
    from playwright.sync_api import Page

SCRAPLING_HINT = 'Install with `pip install "scrapling[fetchers]"` then run `scrapling install`.'
CHECKPOINT_DIR = SCRAPED_DIR / "checkpoints"
DEFAULT_PLATFORMS = ["tiktok", "instagram", "x"]
SOURCE_DOMAINS = {
    "tiktok": ("tiktok.com",),
    "instagram": ("instagram.com",),
    "x": ("x.com", "twitter.com"),
}
DEFAULT_FIELDNAMES = [
    "text",
    "raw_text",
    "source",
    "platform",
    "url",
    "content_id",
    "video_id",
    "author",
    "timestamp",
    "area_hint",
    "city",
    "business_hint",
    "likes",
    "comments",
    "shares",
    "views",
    "hashtags",
    "query",
    "scrape_mode",
    "collected_at",
]


def import_scrapling():
    try:
        from scrapling.fetchers import DynamicSession, FetcherSession
    except ImportError as exc:
        raise RuntimeError(f"Scrapling fetchers are not installed. {SCRAPLING_HINT}") from exc
    return DynamicSession, FetcherSession


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def unix_to_iso(value) -> str:
    try:
        if value in ("", None):
            return ""
        if isinstance(value, str) and not value.isdigit():
            return value
        timestamp = int(float(value))
        if timestamp <= 0:
            return ""
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return ""


def gentle_scroll(page: "Page") -> None:
    for _ in range(4):
        page.wait_for_timeout(600)
        page.mouse.wheel(0, 1800)
    page.wait_for_timeout(900)


def load_platform_cookies(platform: str) -> list[dict]:
    env_key = f"{platform.upper()}_COOKIES_FILE"
    candidates = [
        Path(os.getenv(env_key, "")) if os.getenv(env_key) else None,
        BASE_DIR / "cookies" / f"{platform}.json",
        BASE_DIR / "data" / "cookies" / f"{platform}.json",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return load_cookie_file(candidate)
    return []


def state_path(platform: str) -> Path:
    return CHECKPOINT_DIR / f"{platform}_crawl_state.json"


def load_state(platform: str, output_file: Path) -> dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "platform": platform,
        "schema_version": "scrapling-bootstrap-v1",
        "last_query_index": 0,
        "seen_url_hashes": [],
        "seen_text_hashes": [],
        "discovered_urls": 0,
        "parsed_posts": 0,
        "saved_rows": 0,
        "duplicate_urls": 0,
        "duplicate_texts": 0,
    }
    path = state_path(platform)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            state.update(saved)
        except Exception:
            pass

    existing_rows = load_csv_rows(output_file)
    state["saved_rows"] = len(existing_rows)
    for row in existing_rows:
        url = row.get("url", "")
        text = row.get("text", "")
        if url:
            state["seen_url_hashes"].append(stable_hash(url))
        if text:
            state["seen_text_hashes"].append(stable_hash(normalize_for_dedupe(text)))
    state["seen_url_hashes"] = sorted(set(state["seen_url_hashes"]))
    state["seen_text_hashes"] = sorted(set(state["seen_text_hashes"]))
    return state


def save_state(state: dict) -> None:
    state_path(state["platform"]).parent.mkdir(parents=True, exist_ok=True)
    with open(state_path(state["platform"]), "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def save_rows(output_file: Path, rows: list[dict]) -> None:
    fieldnames = list(DEFAULT_FIELDNAMES)
    if rows:
        for key in rows[0]:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv_rows(output_file, rows, fieldnames)


def search_urls(domain: str, query: str) -> list[str]:
    encoded = quote_plus(f"site:{domain} {query}")
    return [f"https://www.bing.com/search?q={encoded}"]


def clean_candidate_url(url: str, platform: str) -> str:
    canonical = canonicalize_url(url)
    if not canonical:
        return ""
    parsed = urlparse(canonical)
    host = parsed.netloc.lower()
    if not any(domain in host for domain in SOURCE_DOMAINS[platform]):
        return ""

    if platform == "tiktok" and "/video/" not in parsed.path:
        return ""
    if platform == "instagram" and not any(marker in parsed.path for marker in ["/p/", "/reel/"]):
        return ""
    if platform == "x" and "/status/" not in parsed.path:
        return ""
    return canonical


def discover_urls_via_search(session, platform: str, query: str, max_per_query: int) -> list[str]:
    results: list[str] = []
    for domain in SOURCE_DOMAINS[platform]:
        for search_url in search_urls(domain, query):
            try:
                page = session.get(search_url)
            except Exception:
                continue
            for href in page.css("a::attr(href)").getall():
                candidate = clean_candidate_url(href, platform)
                if candidate and candidate not in results:
                    results.append(candidate)
                if len(results) >= max_per_query:
                    return results
    return results


def discover_tiktok_direct(dynamic_session, query: str, max_per_query: int) -> list[str]:
    search_url = f"https://www.tiktok.com/search?q={quote_plus(query)}"
    page = dynamic_session.fetch(
        search_url,
        wait=1000,
        timeout=45000,
        disable_resources=False,
        network_idle=False,
        wait_selector="a",
        page_action=gentle_scroll,
    )
    results: list[str] = []
    for href in page.css("a::attr(href)").getall():
        candidate = clean_candidate_url(href, "tiktok")
        if candidate and candidate not in results:
            results.append(candidate)
        if len(results) >= max_per_query:
            break
    return results


def extract_script_json(page, selector: str) -> dict | list | None:
    raw = page.css(f"{selector}::text").get()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def walk_dicts(payload):
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from walk_dicts(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from walk_dicts(value)


def extract_tiktok_record(dynamic_session, url: str, seed: dict, scrape_mode: str) -> tuple[dict | None, str | None]:
    try:
        page = dynamic_session.fetch(
            url,
            wait=1200,
            timeout=45000,
            disable_resources=False,
            network_idle=False,
            wait_selector="body",
            page_action=gentle_scroll,
        )
    except Exception as exc:
        return None, f"fetch_error: {exc}"
    return parse_tiktok_page(page, url, seed, scrape_mode)


def extract_tiktok_record_static(fetch_session, url: str, seed: dict, scrape_mode: str) -> tuple[dict | None, str | None]:
    try:
        page = fetch_session.get(url)
    except Exception as exc:
        return None, f"fetch_error: {exc}"
    return parse_tiktok_page(page, url, seed, scrape_mode)


def parse_tiktok_page(page, url: str, seed: dict, scrape_mode: str) -> tuple[dict | None, str | None]:
    payloads = [
        extract_script_json(page, "script#__UNIVERSAL_DATA_FOR_REHYDRATION__"),
        extract_script_json(page, "script#SIGI_STATE"),
        extract_script_json(page, "script#__NEXT_DATA__"),
    ]
    item = {}
    for payload in payloads:
        if not payload:
            continue
        candidates = [
            candidate
            for candidate in walk_dicts(payload)
            if isinstance(candidate.get("desc") or candidate.get("description"), str)
            and len((candidate.get("desc") or candidate.get("description") or "").strip()) >= 8
        ]
        if candidates:
            item = max(candidates, key=lambda candidate: len((candidate.get("desc") or candidate.get("description") or "").strip()))
            break

    raw_text = str(item.get("desc") or item.get("description") or "").strip()
    if not raw_text:
        raw_text = page.css('meta[property="og:description"]::attr(content)').get() or ""
    if not raw_text:
        raw_text = page.css('meta[name="description"]::attr(content)').get() or ""
    raw_text = strip_emoji(raw_text.replace("\n", " ").strip())
    text = normalize_text(raw_text)
    if not text:
        return None, "empty_text_after_normalization"

    stats = item.get("stats") if isinstance(item.get("stats"), dict) else {}
    author = item.get("author")
    if isinstance(author, dict):
        author = author.get("uniqueId") or author.get("nickname") or author.get("name") or ""
    elif not isinstance(author, str):
        author = ""
    if not author:
        match = re.search(r"/@([^/]+)/video/", url)
        author = match.group(1) if match else ""

    video_match = re.search(r"/video/(\d+)", url)
    video_id = video_match.group(1) if video_match else str(item.get("id", ""))
    timestamp = item.get("createTimeISO") or unix_to_iso(item.get("createTime"))

    record = {
        "text": text,
        "raw_text": raw_text,
        "source": "tiktok_scrapling",
        "platform": "tiktok",
        "url": canonicalize_url(url),
        "content_id": video_id,
        "video_id": video_id,
        "author": author,
        "timestamp": timestamp,
        "area_hint": seed["area_hint"],
        "city": seed["city"],
        "business_hint": seed["business_hint"],
        "likes": stats.get("diggCount", item.get("diggCount", "")),
        "comments": stats.get("commentCount", item.get("commentCount", "")),
        "shares": stats.get("shareCount", item.get("shareCount", "")),
        "views": stats.get("playCount", item.get("playCount", "")),
        "hashtags": "|".join(extract_hashtags(raw_text)),
        "query": seed["query"],
        "scrape_mode": scrape_mode,
        "collected_at": now_iso(),
    }
    return record, None


def extract_instagram_record(dynamic_session, url: str, seed: dict, scrape_mode: str) -> tuple[dict | None, str | None]:
    try:
        page = dynamic_session.fetch(url, wait=1000, timeout=45000, wait_selector="body", page_action=gentle_scroll)
    except Exception as exc:
        return None, f"fetch_error: {exc}"
    return parse_instagram_page(page, url, seed, scrape_mode)


def extract_instagram_record_static(fetch_session, url: str, seed: dict, scrape_mode: str) -> tuple[dict | None, str | None]:
    try:
        page = fetch_session.get(url)
    except Exception as exc:
        return None, f"fetch_error: {exc}"
    return parse_instagram_page(page, url, seed, scrape_mode)


def parse_instagram_page(page, url: str, seed: dict, scrape_mode: str) -> tuple[dict | None, str | None]:
    raw_text = page.css('meta[property="og:description"]::attr(content)').get() or ""
    if raw_text and ":" in raw_text:
        raw_text = raw_text.split(":", 1)[-1].strip(" \"")
    raw_text = strip_emoji(raw_text)
    if not raw_text:
        return None, "blocked_or_empty_og_description"

    text = normalize_text(raw_text)
    if not text:
        return None, "empty_text_after_normalization"

    title = page.css('meta[property="og:title"]::attr(content)').get() or ""
    post_match = re.search(r"/(?:p|reel)/([^/?#]+)", url)
    record = {
        "text": text,
        "raw_text": raw_text,
        "source": "instagram_scrapling",
        "platform": "instagram",
        "url": canonicalize_url(url),
        "content_id": post_match.group(1) if post_match else "",
        "video_id": "",
        "author": title.split(" on Instagram", 1)[0].strip(),
        "timestamp": "",
        "area_hint": seed["area_hint"],
        "city": seed["city"],
        "business_hint": seed["business_hint"],
        "likes": "",
        "comments": "",
        "shares": "",
        "views": "",
        "hashtags": "|".join(extract_hashtags(raw_text)),
        "query": seed["query"],
        "scrape_mode": scrape_mode,
        "collected_at": now_iso(),
    }
    return record, None


def extract_x_record(dynamic_session, url: str, seed: dict, scrape_mode: str) -> tuple[dict | None, str | None]:
    try:
        page = dynamic_session.fetch(url, wait=800, timeout=45000, wait_selector="body", page_action=gentle_scroll)
    except Exception as exc:
        return None, f"fetch_error: {exc}"
    return parse_x_page(page, url, seed, scrape_mode)


def extract_x_record_static(fetch_session, url: str, seed: dict, scrape_mode: str) -> tuple[dict | None, str | None]:
    try:
        page = fetch_session.get(url)
    except Exception as exc:
        return None, f"fetch_error: {exc}"
    return parse_x_page(page, url, seed, scrape_mode)


def parse_x_page(page, url: str, seed: dict, scrape_mode: str) -> tuple[dict | None, str | None]:
    raw_text = page.css('meta[property="og:description"]::attr(content)').get() or ""
    raw_text = re.sub(r"^\s*[^:]+ on X:\s*", "", raw_text).strip()
    raw_text = strip_emoji(raw_text)
    if not raw_text:
        return None, "blocked_or_empty_og_description"

    text = normalize_text(raw_text)
    if not text:
        return None, "empty_text_after_normalization"

    author_match = re.search(r"https?://(?:www\.)?(?:x|twitter)\.com/([^/]+)/status/", url)
    status_match = re.search(r"/status/(\d+)", url)
    record = {
        "text": text,
        "raw_text": raw_text,
        "source": "x_scrapling",
        "platform": "x",
        "url": canonicalize_url(url),
        "content_id": status_match.group(1) if status_match else "",
        "video_id": "",
        "author": author_match.group(1) if author_match else "",
        "timestamp": "",
        "area_hint": seed["area_hint"],
        "city": seed["city"],
        "business_hint": seed["business_hint"],
        "likes": "",
        "comments": "",
        "shares": "",
        "views": "",
        "hashtags": "|".join(extract_hashtags(raw_text)),
        "query": seed["query"],
        "scrape_mode": scrape_mode,
        "collected_at": now_iso(),
    }
    return record, None


def collect_platform(platform: str, seeds: list[dict], args, manifest: dict) -> None:
    DynamicSession, FetcherSession = import_scrapling()
    output_file = SOCIAL_DIR / f"{platform}_data.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = load_csv_rows(output_file)
    state = load_state(platform, output_file)
    seen_url_hashes = set(state["seen_url_hashes"])
    seen_text_hashes = set(state["seen_text_hashes"])

    cookies = load_platform_cookies(platform)
    scrape_mode = "cookie_session" if cookies else "public_first"

    platform_manifest = manifest["platforms"].setdefault(
        platform,
        {
            "queries": state.get("last_query_index", 0),
            "discovered_urls": state.get("discovered_urls", 0),
            "parsed_posts": state.get("parsed_posts", 0),
            "saved_rows": len(rows),
            "duplicate_urls": state.get("duplicate_urls", 0),
            "duplicate_texts": state.get("duplicate_texts", 0),
            "blocked_reasons": [],
        },
    )

    dynamic_context = (
        nullcontext(None)
        if args.search_engine_only
        else DynamicSession(
            headless=args.headless,
            disable_resources=False,
            network_idle=False,
            real_chrome=False,
            cookies=cookies or None,
        )
    )

    with dynamic_context as dynamic_session, FetcherSession(impersonate="chrome") as fetch_session:
        for query_index, seed in enumerate(seeds[state["last_query_index"] :], start=state["last_query_index"]):
            platform_manifest["queries"] += 1
            if platform == "tiktok":
                if args.search_engine_only:
                    urls = []
                    effective_mode = f"{scrape_mode}+search_engine_only"
                else:
                    try:
                        urls = discover_tiktok_direct(dynamic_session, seed["query"], args.max_per_query)
                        effective_mode = f"{scrape_mode}+direct_search"
                    except Exception:
                        urls = []
                        effective_mode = f"{scrape_mode}+search_engine_fallback"
                if len(urls) < min(3, args.max_per_query):
                    urls = urls + [
                        url
                        for url in discover_urls_via_search(fetch_session, platform, seed["query"], args.max_per_query)
                        if url not in urls
                    ]
                    effective_mode = f"{scrape_mode}+search_engine_fallback"
                extractor = extract_tiktok_record_static if args.search_engine_only else extract_tiktok_record
            else:
                urls = discover_urls_via_search(fetch_session, platform, seed["query"], args.max_per_query)
                effective_mode = f"{scrape_mode}+search_engine"
                if platform == "instagram":
                    extractor = extract_instagram_record_static if args.search_engine_only else extract_instagram_record
                else:
                    extractor = extract_x_record_static if args.search_engine_only else extract_x_record

            if not urls:
                platform_manifest["blocked_reasons"].append({"query": seed["query"], "reason": "no_public_urls_found"})
                state["last_query_index"] = query_index + 1
                save_state(state)
                continue

            for discovered_index, url in enumerate(urls[: args.max_per_query], start=1):
                url_hash = stable_hash(url)
                platform_manifest["discovered_urls"] += 1
                state["discovered_urls"] += 1
                if url_hash in seen_url_hashes:
                    platform_manifest["duplicate_urls"] += 1
                    state["duplicate_urls"] += 1
                    continue

                record, blocked_reason = extractor(dynamic_session, url, seed, effective_mode)
                seen_url_hashes.add(url_hash)
                if blocked_reason:
                    platform_manifest["blocked_reasons"].append({"url": url, "reason": blocked_reason})
                if not record:
                    continue

                text_hash = stable_hash(normalize_for_dedupe(record["text"]))
                if text_hash in seen_text_hashes:
                    platform_manifest["duplicate_texts"] += 1
                    state["duplicate_texts"] += 1
                    continue

                seen_text_hashes.add(text_hash)
                rows.append(record)
                platform_manifest["parsed_posts"] += 1
                platform_manifest["saved_rows"] = len(rows)
                state["parsed_posts"] += 1
                state["saved_rows"] = len(rows)

                if state["discovered_urls"] % 100 == 0 or state["parsed_posts"] % 250 == 0:
                    state["seen_url_hashes"] = sorted(seen_url_hashes)
                    state["seen_text_hashes"] = sorted(seen_text_hashes)
                    save_rows(output_file, rows)
                    save_state(state)

                if args.dry_run and len(rows) >= args.dry_run_target:
                    break
                if args.max_saved_rows and len(rows) >= args.max_saved_rows:
                    break

            state["last_query_index"] = query_index + 1
            state["seen_url_hashes"] = sorted(seen_url_hashes)
            state["seen_text_hashes"] = sorted(seen_text_hashes)
            save_rows(output_file, rows)
            save_state(state)

            if args.dry_run and len(rows) >= args.dry_run_target:
                break
            if args.max_saved_rows and len(rows) >= args.max_saved_rows:
                break
            time.sleep(args.query_delay)


def maybe_refresh_google_maps(args, manifest: dict) -> None:
    gmaps_file = SOCIAL_DIR / "gmaps_reviews.csv"
    existing = len(load_csv_rows(gmaps_file))
    manifest["platforms"]["google_maps"] = {
        "queries": 0,
        "discovered_urls": 0,
        "parsed_posts": existing,
        "saved_rows": existing,
        "duplicate_urls": 0,
        "duplicate_texts": 0,
        "blocked_reasons": [],
    }
    if not args.refresh_gmaps:
        return

    from collect_gmaps_reviews import main as collect_gmaps_main

    collect_gmaps_main()
    refreshed = len(load_csv_rows(gmaps_file))
    manifest["platforms"]["google_maps"]["parsed_posts"] = refreshed
    manifest["platforms"]["google_maps"]["saved_rows"] = refreshed


def write_manual_review_samples(signal_rows: list[dict[str, str]], ner_rows: list[dict[str, object]]) -> None:
    tiktok_rows = [row for row in signal_rows if row["platform"] == "tiktok"][:100]
    ner_sample = [row for row in ner_rows if row["candidate_spans"]][:50]
    if tiktok_rows:
        write_csv_rows(
            SCRAPED_DIR / "review_tiktok_sample.csv",
            tiktok_rows,
            ["text", "platform", "url", "city", "area_hint", "business_hint", "source"],
        )
    if ner_sample:
        sample_path = SCRAPED_DIR / "review_ner_sample.jsonl"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sample_path, "w", encoding="utf-8") as handle:
            for row in ner_sample:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rebuild_bootstrap_outputs(manifest: dict) -> None:
    signal_rows = build_signal_bootstrap_rows()
    signal_output = SCRAPED_DIR / "signal_bootstrap.csv"
    write_csv_rows(
        signal_output,
        signal_rows,
        ["text", "source", "platform", "url", "timestamp", "area_hint", "city", "business_hint", "query", "provenance_split"],
    )

    ner_rows = build_ner_bootstrap_rows(signal_rows)
    ner_output = SCRAPED_DIR / "ner_bootstrap.jsonl"
    ner_output.parent.mkdir(parents=True, exist_ok=True)
    with open(ner_output, "w", encoding="utf-8") as handle:
        for row in ner_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_manual_review_samples(signal_rows, ner_rows)
    manifest["source_counts"] = {
        platform: details.get("saved_rows", 0)
        for platform, details in manifest["platforms"].items()
    }
    manifest["source_counts"]["signal_bootstrap"] = len(signal_rows)
    manifest["source_counts"]["ner_bootstrap"] = len(ner_rows)


def parse_args(argv=None, default_platforms=None):
    parser = argparse.ArgumentParser(description="Scrapling-first social bootstrap collector.")
    parser.add_argument("--platform", dest="platforms", action="append", choices=DEFAULT_PLATFORMS, help="Platform to scrape.")
    parser.add_argument("--max-queries", type=int, default=18, help="Maximum query seeds to process.")
    parser.add_argument("--max-per-query", type=int, default=12, help="Maximum discovered URLs per query.")
    parser.add_argument("--max-saved-rows", type=int, default=0, help="Stop after saving this many rows for the current platform. 0 disables the limit.")
    parser.add_argument("--dry-run", action="store_true", help="Run a short smoke test instead of a long scrape.")
    parser.add_argument("--dry-run-target", type=int, default=100, help="Stop after this many saved rows in dry-run mode.")
    parser.add_argument("--headless", action="store_true", default=False, help="Run browsers headless.")
    parser.add_argument("--query-delay", type=float, default=1.2, help="Delay in seconds between queries.")
    parser.add_argument("--refresh-gmaps", action="store_true", help="Refresh Google Maps reviews using the existing API collector.")
    parser.add_argument("--search-engine-only", action="store_true", help="Skip the browser layer and use Scrapling HTTP fetchers only.")
    args = parser.parse_args(argv)
    args.platforms = args.platforms or list(default_platforms or DEFAULT_PLATFORMS)
    if args.dry_run:
        args.max_queries = min(args.max_queries, 4)
        args.max_per_query = min(args.max_per_query, 6)
    return args


def main(argv=None, default_platforms=None):
    load_dotenv(BASE_DIR / ".env")
    args = parse_args(argv=argv, default_platforms=default_platforms)
    SOCIAL_DIR.mkdir(parents=True, exist_ok=True)
    SCRAPED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "scrapling-bootstrap-v1",
        "generated_at": now_iso(),
        "collector_mode": {
            "public_first": True,
            "cookie_auto": True,
            "playwright_page_action": True,
            "scrapling_required": True,
        },
        "platforms": {},
        "blocked_sources": [],
    }

    seeds = iter_query_seeds(max_queries=args.max_queries)
    maybe_refresh_google_maps(args, manifest)

    for platform in args.platforms:
        collect_platform(platform, seeds, args, manifest)

    rebuild_bootstrap_outputs(manifest)

    for platform, details in manifest["platforms"].items():
        for blocked in details.get("blocked_reasons", []):
            manifest["blocked_sources"].append({"platform": platform, **blocked})

    dump_json(SCRAPED_DIR / "manifest.json", manifest)

    print("=" * 70)
    print(" Scrapling Social Bootstrap Collection Complete")
    print("=" * 70)
    for platform, details in manifest["platforms"].items():
        print(f"  {platform:12s} rows={details.get('saved_rows', 0):>5} blocked={len(details.get('blocked_reasons', []))}")
    print(f"  signal_bootstrap: {manifest['source_counts']['signal_bootstrap']}")
    print(f"  ner_bootstrap   : {manifest['source_counts']['ner_bootstrap']}")
    print(f"  manifest        : {SCRAPED_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()

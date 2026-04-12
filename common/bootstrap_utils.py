from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from common.market_catalog import BUSINESS_HINTS, CITIES_KECAMATAN
from common.text_normalization import (
    is_probably_indonesian,
    normalize_for_dedupe,
    normalize_text,
    strip_emoji,
    tokenize_with_offsets,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SCRAPED_DIR = BASE_DIR / "data" / "scraped"
SOCIAL_DIR = BASE_DIR / "data" / "social_media"
POI_FILE = BASE_DIR / "data" / "poi" / "overpass_poi.csv"
ADMIN_DIR = BASE_DIR / "data" / "geospatial" / "Wilayah-Administratif-Indonesia" / "csv"
SENTENCE_BOUNDARY_RE = re.compile(r"(?:[\r\n]+|(?<=[.!?])\s+)")
CLAUSE_BOUNDARY_RE = re.compile(r"\s*[;,]\s+")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_split(text: str) -> str:
    value = int(stable_hash(text)[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "validation"
    return "test"


def text_candidates(raw_text: str) -> list[str]:
    """
    Expand a noisy post or review into classifier-ready candidates.

    We keep the full text when it is reasonably short, and also salvage
    sentence/clause-level Indonesian fragments from longer mixed-language text.
    """
    if not isinstance(raw_text, str):
        return []

    raw_text = strip_emoji(raw_text).replace("\u200b", " ").strip()
    if not raw_text:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(candidate_text: str) -> None:
        normalized = normalize_text(candidate_text)
        token_count = len(normalized.split())
        if len(normalized) < 12 or len(normalized) > 240:
            return
        if token_count < 3 or token_count > 48:
            return
        dedupe_key = normalize_for_dedupe(normalized)
        if not dedupe_key or dedupe_key in seen:
            return
        seen.add(dedupe_key)
        candidates.append(normalized)

    whole_text = normalize_text(raw_text)
    if len(whole_text) <= 220:
        add_candidate(whole_text)

    for sentence in SENTENCE_BOUNDARY_RE.split(raw_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        add_candidate(sentence)
        if len(sentence) > 160:
            for clause in CLAUSE_BOUNDARY_RE.split(sentence):
                add_candidate(clause.strip())

    return candidates[:8]


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    raw = url.strip()
    if raw.startswith("//"):
        raw = f"https:{raw}"
    parsed = urlparse(raw)
    if not parsed.scheme:
        raw = f"https://{raw.lstrip('/')}"
        parsed = urlparse(raw)

    query = parse_qs(parsed.query)
    if "uddg" in query:
        return canonicalize_url(unquote(query["uddg"][0]))

    host = parsed.netloc.lower().replace("m.", "").replace("mobile.", "")
    path = re.sub(r"/+$", "", parsed.path or "/")
    return f"{parsed.scheme or 'https'}://{host}{path}"


def google_maps_search_url(place_name: str, place_address: str) -> str:
    query = " ".join(part for part in [place_name, place_address] if part).strip()
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}" if query else ""


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_cookie_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []

    if isinstance(payload, dict):
        cookies = payload.get("cookies", [])
        return cookies if isinstance(cookies, list) else []
    return payload if isinstance(payload, list) else []


@dataclass
class CandidateSpan:
    text: str
    start_char: int
    end_char: int
    label: str
    source: str


class GazetteerMatcher:
    def __init__(self) -> None:
        self.city_terms = {city.lower(): city for city in CITIES_KECAMATAN}
        self.district_terms = {
            district.lower(): {"district": district, "city": city}
            for city, districts in CITIES_KECAMATAN.items()
            for district in districts
        }
        self.business_terms = {term.lower(): term for term in BUSINESS_HINTS}
        self.poi_by_city = self._load_poi_by_city()

    def _load_poi_by_city(self) -> dict[str, list[str]]:
        if not POI_FILE.exists():
            return {}
        import pandas as pd

        poi_df = pd.read_csv(POI_FILE)
        poi_df = poi_df.dropna(subset=["name", "city"])
        grouped: dict[str, list[str]] = {}
        for city, city_df in poi_df.groupby("city"):
            names = sorted(
                {
                    str(name).strip()
                    for name in city_df["name"].tolist()
                    if isinstance(name, str) and len(name.strip()) >= 4
                },
                key=len,
                reverse=True,
            )
            grouped[str(city).strip().lower()] = names[:500]
        return grouped

    @staticmethod
    def _find_term_positions(text: str, term: str) -> Iterable[tuple[int, int]]:
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
        for match in pattern.finditer(text):
            yield match.start(), match.end()

    def extract_candidates(self, text: str, city_hint: str = "", area_hint: str = "") -> list[CandidateSpan]:
        lowered = text.lower()
        spans: list[CandidateSpan] = []
        seen: set[tuple[int, int, str]] = set()

        def add_spans(label: str, source: str, term: str) -> None:
            for start, end in self._find_term_positions(lowered, term.lower()):
                key = (start, end, label)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(CandidateSpan(text=text[start:end], start_char=start, end_char=end, label=label, source=source))

        for city_term in self.city_terms:
            add_spans("LOC_CITY", "city_catalog", city_term)

        for district_term in self.district_terms:
            add_spans("LOC_DISTRICT", "district_catalog", district_term)

        for business_term in self.business_terms:
            add_spans("BIZ_TYPE", "business_catalog", business_term)

        city_key = city_hint.lower().strip()
        if city_key in self.poi_by_city:
            for poi_name in self.poi_by_city[city_key]:
                add_spans("POI_NAME", "overpass_poi", poi_name.lower())

        if area_hint:
            add_spans("LOC_DISTRICT", "row_area_hint", area_hint.lower())

        return sorted(spans, key=lambda span: (span.start_char, span.end_char))


def weak_ner_tags_from_candidates(text: str, candidate_spans: list[CandidateSpan]) -> tuple[list[str], list[str]]:
    token_data = tokenize_with_offsets(text)
    tokens = [token for token, _, _ in token_data]
    labels = ["O"] * len(tokens)

    for span in candidate_spans:
        if span.label.startswith("LOC_"):
            prefix = "LOC"
        else:
            prefix = "ORG"
        matched = [index for index, (_, start, end) in enumerate(token_data) if start >= span.start_char and end <= span.end_char]
        if not matched:
            continue
        labels[matched[0]] = f"B-{prefix}"
        for index in matched[1:]:
            labels[index] = f"I-{prefix}"
    return tokens, labels


def build_signal_bootstrap_rows(*, include_google_maps: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_texts: set[str] = set()

    def append_row(
        raw_text: str,
        platform: str,
        source: str,
        url: str,
        timestamp: str,
        city: str,
        area_hint: str,
        business_hint: str,
        query: str,
        query_intent: str = "",
    ) -> None:
        strict_language = platform == "google_maps"
        for text in text_candidates(raw_text):
            if not is_probably_indonesian(text, strict=strict_language):
                continue
            text_hash = stable_hash(normalize_for_dedupe(text))
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)
            rows.append(
                {
                    "text": text,
                    "source": source,
                    "platform": platform,
                    "url": url,
                    "timestamp": timestamp,
                    "area_hint": area_hint,
                    "city": city,
                    "business_hint": business_hint,
                    "query": normalize_text(query),
                    "query_intent": query_intent,
                    "provenance_split": stable_split(text),
                }
            )

    for platform_file, platform_name in [
        (SOCIAL_DIR / "tiktok_data.csv", "tiktok"),
        (SOCIAL_DIR / "instagram_data.csv", "instagram"),
        (SOCIAL_DIR / "x_data.csv", "x"),
    ]:
        for row in load_csv_rows(platform_file):
            append_row(
                raw_text=row.get("raw_text") or row.get("text", ""),
                platform=platform_name,
                source=row.get("source", platform_name),
                url=row.get("url", ""),
                timestamp=row.get("timestamp", ""),
                city=row.get("city", ""),
                area_hint=row.get("area_hint", ""),
                business_hint=row.get("business_hint", ""),
                query=row.get("query", ""),
                query_intent=row.get("query_intent", ""),
            )

    if include_google_maps:
        for row in load_csv_rows(SOCIAL_DIR / "gmaps_reviews.csv"):
            append_row(
                raw_text=row.get("text", ""),
                platform="google_maps",
                source=row.get("source", "google_maps"),
                url=google_maps_search_url(row.get("place_name", ""), row.get("place_address", "")),
                timestamp=row.get("timestamp", ""),
                city=row.get("city", ""),
                area_hint=row.get("area_hint", ""),
                business_hint=row.get("business_hint", ""),
                query=row.get("query", ""),
                query_intent=row.get("query_intent", ""),
            )

    return rows


def build_ner_bootstrap_rows(signal_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    matcher = GazetteerMatcher()
    bootstrap_rows: list[dict[str, object]] = []
    for row in signal_rows:
        candidate_spans = matcher.extract_candidates(
            text=row["text"],
            city_hint=row.get("city", ""),
            area_hint=row.get("area_hint", ""),
        )
        tokens, weak_ner_tags = weak_ner_tags_from_candidates(row["text"], candidate_spans)
        bootstrap_rows.append(
            {
                "text": row["text"],
                "tokens": tokens,
                "candidate_spans": [candidate.__dict__ for candidate in candidate_spans],
                "weak_ner_tags": weak_ner_tags,
                "source": row["source"],
                "platform": row["platform"],
                "url": row["url"],
                "city": row.get("city", ""),
                "area_hint": row.get("area_hint", ""),
                "business_hint": row.get("business_hint", ""),
            }
        )
    return bootstrap_rows

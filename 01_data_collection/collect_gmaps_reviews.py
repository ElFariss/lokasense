#!/usr/bin/env python3
"""
Collect Google Maps reviews with the Places API (New).

This script is intentionally opt-in because it can incur API charges.
The maintained notebook and Scrapling bootstrap do not call it unless the
user explicitly enables a billable refresh.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.text_normalization import language_scores

load_dotenv(BASE_DIR / ".env")

OUTPUT_DIR = BASE_DIR / "data" / "social_media"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("GOOGLE_MAP", "")

TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places"

CITIES_KECAMATAN = {
    "Malang": ["Lowokwaru", "Klojen", "Blimbing", "Sukun", "Kedungkandang"],
    "Surabaya": ["Gubeng", "Tegalsari", "Genteng", "Wonokromo", "Rungkut"],
    "Yogyakarta": ["Gondokusuman", "Umbulharjo", "Kotagede", "Mergangsan", "Danurejan"],
    "Bandung": ["Coblong", "Bandung Wetan", "Sumur Bandung", "Cicendo", "Lengkong"],
    "Semarang": ["Semarang Tengah", "Semarang Selatan", "Candisari", "Gajahmungkur", "Banyumanik"],
}
BUSINESS_TYPES = ["ayam geprek", "kopi", "mie", "laundry", "kedai makan"]

SEARCH_QUERIES = []
for city, kecamatan_list in CITIES_KECAMATAN.items():
    for kecamatan in kecamatan_list:
        for business_type in BUSINESS_TYPES:
            SEARCH_QUERIES.append(
                {
                    "query": f"{business_type} di {kecamatan} {city}",
                    "business_type": business_type,
                    "city": city,
                    "kecamatan": kecamatan,
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Billable Google Maps review collector.")
    parser.add_argument(
        "--confirm-billable",
        action="store_true",
        help="Required because this script can incur Google Maps API charges.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=len(SEARCH_QUERIES),
        help="Limit how many predefined search queries are executed.",
    )
    parser.add_argument(
        "--keep-all-languages",
        action="store_true",
        help="Disable the default Indonesian-only filtering and keep all review languages.",
    )
    return parser.parse_args(argv)


def classify_review_language(text: str) -> str:
    scores = language_scores(str(text))
    id_score = scores["id"] + scores["slang"]
    en_score = scores["en"]
    if id_score == 0 and en_score == 0:
        return "unclear"
    if id_score >= en_score:
        return "indonesian_dominant"
    return "english_dominant"


def search_places(query: str, api_key: str) -> list[dict]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.location,places.types",
    }
    body = {"textQuery": query, "languageCode": "id", "maxResultCount": 10}
    try:
        response = requests.post(TEXT_SEARCH_URL, headers=headers, json=body, timeout=30)
        if response.status_code == 200:
            return response.json().get("places", [])
        print(f"  Search {response.status_code}: {response.text[:150]}")
        return []
    except Exception as exc:
        print(f"  Search failed: {exc}")
        return []


def get_reviews(place_id: str, api_key: str) -> list[dict]:
    url = f"{PLACE_DETAILS_URL}/{place_id}"
    headers = {"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": "reviews"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json().get("reviews", [])
        return []
    except Exception:
        return []


def save_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=" * 70)
    print(" Google Maps Reviews Collection — explicit opt-in")
    print("=" * 70)

    if not args.confirm_billable:
        print("Refusing to run without --confirm-billable.")
        print("This collector uses the Google Maps Places API and can incur charges.")
        return

    if not API_KEY:
        print("GOOGLE_MAP not found in .env.")
        return

    all_reviews: list[dict[str, object]] = []
    places_seen: set[str] = set()
    stats = {"queries": 0, "places": 0, "reviews": 0, "filtered_non_indonesian": 0, "errors": 0}
    queries = SEARCH_QUERIES[: max(0, min(args.max_queries, len(SEARCH_QUERIES)))]
    total = len(queries)
    print(f"  {total} search queries to process")
    print(f"  Indonesian-only filtering: {not args.keep_all_languages}")
    print()

    for index, seed in enumerate(queries, start=1):
        query = seed["query"]
        print(f"  [{index}/{total}] {query}", end="")
        places = search_places(query, API_KEY)
        stats["queries"] += 1
        new_places = 0

        for place in places:
            place_id = place.get("id", "")
            if not place_id or place_id in places_seen:
                continue
            places_seen.add(place_id)
            new_places += 1
            stats["places"] += 1

            name = place.get("displayName", {}).get("text", "")
            address = place.get("formattedAddress", "")
            rating = place.get("rating", 0)
            rating_count = place.get("userRatingCount", 0)
            location = place.get("location", {})
            lat, lng = location.get("latitude", ""), location.get("longitude", "")

            for review in get_reviews(place_id, API_KEY):
                text = str(review.get("text", {}).get("text", "")).strip()
                if len(text) < 10:
                    continue
                review_language = classify_review_language(text)
                if not args.keep_all_languages and review_language != "indonesian_dominant":
                    stats["filtered_non_indonesian"] += 1
                    continue

                stats["reviews"] += 1
                all_reviews.append(
                    {
                        "text": text,
                        "source": "google_maps",
                        "timestamp": review.get("publishTime", ""),
                        "area_hint": seed["kecamatan"],
                        "city": seed["city"],
                        "business_hint": seed["business_type"],
                        "query": seed["query"],
                        "place_name": name,
                        "place_address": address,
                        "place_rating": rating,
                        "place_rating_count": rating_count,
                        "review_rating": review.get("rating", 0),
                        "review_language": review_language,
                        "lat": lat,
                        "lng": lng,
                    }
                )
            time.sleep(0.3)

        print(f" -> {new_places} new places, {len(all_reviews)} kept reviews")
        time.sleep(0.5)

        if index % 25 == 0 and all_reviews:
            save_rows(all_reviews, OUTPUT_DIR / "gmaps_reviews_checkpoint.csv")

    output_file = OUTPUT_DIR / "gmaps_reviews.csv"
    save_rows(all_reviews, output_file)

    with open(LOG_DIR / "gmaps_collection_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print()
    print("=" * 70)
    print(f" Done: {stats['reviews']} reviews kept from {stats['places']} places")
    print(f" Filtered non-Indonesian reviews: {stats['filtered_non_indonesian']}")
    print(f" Saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

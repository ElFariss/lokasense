#!/usr/bin/env python3
"""
Collect Google Maps reviews via Places API (New).
Uses the user's GCloud credentials to fetch reviews for target business types
in target kecamatan across Indonesian cities.

Requires:
    - GOOGLE_MAPS_API_KEY in .env OR Application Default Credentials via gcloud
    - Places API (New) enabled in Google Cloud Console

Usage:
    python scripts/collect_gmaps_reviews.py
"""
import os
import json
import time
import csv
from pathlib import Path
from dotenv import load_dotenv
import requests

# ─── Configuration ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
ENV_DIR = BASE_DIR.parent  # UGM_HACKATHON level where .env lives
load_dotenv(ENV_DIR / ".env")

OUTPUT_DIR = BASE_DIR / "data" / "social_media"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Try to get API key from environment
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# Places API (New) endpoints
TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places"

# Target search queries: (business_type, city, kecamatan_hint)
SEARCH_QUERIES = []
CITIES_KECAMATAN = {
    "Malang": ["Lowokwaru", "Klojen", "Blimbing", "Sukun", "Kedungkandang"],
    "Surabaya": ["Gubeng", "Tegalsari", "Genteng", "Wonokromo", "Rungkut"],
    "Yogyakarta": ["Gondokusuman", "Umbulharjo", "Kotagede", "Mergangsan", "Danurejan"],
    "Bandung": ["Coblong", "Bandung Wetan", "Sumur Bandung", "Cicendo", "Lengkong"],
    "Semarang": ["Semarang Tengah", "Semarang Selatan", "Candisari", "Gajahmungkur", "Banyumanik"],
}
BUSINESS_TYPES = ["ayam geprek", "kopi", "mie", "laundry", "kedai makan"]

for city, kecamatan_list in CITIES_KECAMATAN.items():
    for kec in kecamatan_list:
        for biz in BUSINESS_TYPES:
            SEARCH_QUERIES.append({
                "query": f"{biz} di {kec} {city}",
                "business_type": biz,
                "city": city,
                "kecamatan": kec,
            })


def search_places(query: str, api_key: str) -> list:
    """Search for places using Places API (New) Text Search."""
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.location,places.types",
    }
    body = {
        "textQuery": query,
        "languageCode": "id",
        "maxResultCount": 10,
    }

    try:
        response = requests.post(TEXT_SEARCH_URL, headers=headers, json=body, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("places", [])
        else:
            print(f"  ⚠ Search error {response.status_code}: {response.text[:200]}")
            return []
    except Exception as e:
        print(f"  ⚠ Search exception: {e}")
        return []


def get_place_reviews(place_id: str, api_key: str) -> list:
    """Get reviews for a specific place using Place Details (New)."""
    url = f"{PLACE_DETAILS_URL}/{place_id}"
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "reviews",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("reviews", [])
        else:
            print(f"    ⚠ Details error {response.status_code}: {response.text[:200]}")
            return []
    except Exception as e:
        print(f"    ⚠ Details exception: {e}")
        return []


def main():
    """Main collection loop."""
    print("=" * 70)
    print(" Google Maps Reviews Collection — LokaSense / Pasarint")
    print("=" * 70)

    if not GOOGLE_MAPS_API_KEY:
        print("\n⚠ GOOGLE_MAPS_API_KEY not found in .env")
        print("  Please add it to your .env file:")
        print("  GOOGLE_MAPS_API_KEY=your_api_key_here")
        print("\n  To get an API key:")
        print("  1. Go to https://console.cloud.google.com")
        print("  2. Enable 'Places API (New)'")
        print("  3. Create API key under 'Credentials'")
        print("  4. Add to .env file")
        return

    all_reviews = []
    places_seen = set()  # deduplicate by place_id

    total_queries = len(SEARCH_QUERIES)
    print(f"\n  Total search queries: {total_queries}")
    print(f"  Estimated time: ~{total_queries * 2}s (with rate limiting)\n")

    for i, sq in enumerate(SEARCH_QUERIES):
        query = sq["query"]
        print(f"  [{i+1}/{total_queries}] Searching: {query}")

        places = search_places(query, GOOGLE_MAPS_API_KEY)
        print(f"    Found {len(places)} places")

        for place in places:
            place_id = place.get("id", "")
            if place_id in places_seen:
                continue
            places_seen.add(place_id)

            display_name = place.get("displayName", {}).get("text", "")
            address = place.get("formattedAddress", "")
            rating = place.get("rating", 0)
            rating_count = place.get("userRatingCount", 0)
            location = place.get("location", {})
            lat = location.get("latitude", "")
            lng = location.get("longitude", "")

            # Get reviews for this place
            reviews = get_place_reviews(place_id, GOOGLE_MAPS_API_KEY)
            print(f"    → {display_name}: {len(reviews)} reviews")

            for review in reviews:
                review_text = review.get("text", {}).get("text", "")
                if not review_text or len(review_text.strip()) < 10:
                    continue

                review_rating = review.get("rating", 0)
                review_time = review.get("publishTime", "")
                review_lang = review.get("originalText", {}).get("languageCode", "")

                all_reviews.append({
                    "text": review_text.strip(),
                    "source": "google_maps",
                    "timestamp": review_time,
                    "area_hint": sq["kecamatan"],
                    "city": sq["city"],
                    "business_hint": sq["business_type"],
                    "place_name": display_name,
                    "place_address": address,
                    "place_rating": rating,
                    "place_rating_count": rating_count,
                    "review_rating": review_rating,
                    "lat": lat,
                    "lng": lng,
                    "language": review_lang,
                })

            time.sleep(0.5)  # rate limiting between detail calls

        time.sleep(1.0)  # rate limiting between search calls

        # Checkpoint save every 25 queries
        if (i + 1) % 25 == 0:
            _save_checkpoint(all_reviews, OUTPUT_DIR)

    # Final save
    output_file = OUTPUT_DIR / "gmaps_reviews.csv"
    if all_reviews:
        fieldnames = all_reviews[0].keys()
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_reviews)

    print(f"\n{'=' * 70}")
    print(f" DONE: {len(all_reviews)} reviews from {len(places_seen)} places")
    print(f" Saved to: {output_file}")
    print(f"{'=' * 70}")


def _save_checkpoint(reviews, output_dir):
    """Save intermediate results as checkpoint."""
    if not reviews:
        return
    checkpoint_file = output_dir / "gmaps_reviews_checkpoint.csv"
    fieldnames = reviews[0].keys()
    with open(checkpoint_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)
    print(f"    📌 Checkpoint: {len(reviews)} reviews saved")


if __name__ == "__main__":
    main()

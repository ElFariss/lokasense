#!/usr/bin/env python3
"""
01_data_collection/collect_gmaps_reviews.py
Collect Google Maps reviews via Places API (New).

Requires: GOOGLE_MAP in .env
Usage:    python 01_data_collection/collect_gmaps_reviews.py
"""
import os, json, time, csv
from pathlib import Path
from dotenv import load_dotenv
import requests

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR.parent / ".env")

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
for city, kec_list in CITIES_KECAMATAN.items():
    for kec in kec_list:
        for biz in BUSINESS_TYPES:
            SEARCH_QUERIES.append({"query": f"{biz} di {kec} {city}", "business_type": biz, "city": city, "kecamatan": kec})


def search_places(query, api_key):
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.location,places.types",
    }
    body = {"textQuery": query, "languageCode": "id", "maxResultCount": 10}
    try:
        r = requests.post(TEXT_SEARCH_URL, headers=headers, json=body, timeout=30)
        if r.status_code == 200:
            return r.json().get("places", [])
        else:
            print(f"  Search {r.status_code}: {r.text[:150]}")
            return []
    except Exception as e:
        print(f"  {e}")
        return []


def get_reviews(place_id, api_key):
    url = f"{PLACE_DETAILS_URL}/{place_id}"
    headers = {"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": "reviews"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.json().get("reviews", [])
        else:
            return []
    except:
        return []


def main():
    print("=" * 70)
    print(" Google Maps Reviews Collection — LokaSense")
    print("=" * 70)

    if not API_KEY:
        print("GOOGLE_MAP not found in .env.")
        return

    all_reviews = []
    places_seen = set()
    stats = {"queries": 0, "places": 0, "reviews": 0, "errors": 0}
    total = len(SEARCH_QUERIES)
    print(f"  {total} search queries to process\n")

    for i, sq in enumerate(SEARCH_QUERIES):
        query = sq["query"]
        print(f"  [{i+1}/{total}] {query}", end="")

        places = search_places(query, API_KEY)
        stats["queries"] += 1
        new_places = 0

        for place in places:
            pid = place.get("id", "")
            if pid in places_seen:
                continue
            places_seen.add(pid)
            new_places += 1
            stats["places"] += 1

            name = place.get("displayName", {}).get("text", "")
            addr = place.get("formattedAddress", "")
            rating = place.get("rating", 0)
            rating_count = place.get("userRatingCount", 0)
            loc = place.get("location", {})
            lat, lng = loc.get("latitude", ""), loc.get("longitude", "")

            reviews = get_reviews(pid, API_KEY)
            for rev in reviews:
                text = rev.get("text", {}).get("text", "")
                if not text or len(text.strip()) < 10:
                    continue
                stats["reviews"] += 1
                all_reviews.append({
                    "text": text.strip(),
                    "source": "google_maps",
                    "timestamp": rev.get("publishTime", ""),
                    "area_hint": sq["kecamatan"],
                    "city": sq["city"],
                    "business_hint": sq["business_type"],
                    "place_name": name,
                    "place_address": addr,
                    "place_rating": rating,
                    "place_rating_count": rating_count,
                    "review_rating": rev.get("rating", 0),
                    "lat": lat,
                    "lng": lng,
                })
            time.sleep(0.3)

        print(f" → {new_places} new places, {len(all_reviews)} total reviews")
        time.sleep(0.5)

        if (i + 1) % 25 == 0 and all_reviews:
            _save(all_reviews, OUTPUT_DIR / "gmaps_reviews_checkpoint.csv")

    # Final save
    output_file = OUTPUT_DIR / "gmaps_reviews.csv"
    _save(all_reviews, output_file)

    # Save collection stats
    with open(LOG_DIR / "gmaps_collection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f" DONE: {stats['reviews']} reviews from {stats['places']} places")
    print(f" Saved to: {output_file}")
    print(f"{'=' * 70}")


def _save(reviews, path):
    if not reviews:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reviews[0].keys())
        writer.writeheader()
        writer.writerows(reviews)


if __name__ == "__main__":
    main()

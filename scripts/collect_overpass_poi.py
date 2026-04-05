#!/usr/bin/env python3
"""
Collect POI (Points of Interest) data from OpenStreetMap via Overpass API.
This is the Business Presence Engine data source — counts actual businesses per area.

Free, no API key needed, no rate limit concerns (just be polite with delays).

Usage:
    python scripts/collect_overpass_poi.py
"""
import requests
import json
import time
import csv
from pathlib import Path
from collections import Counter

# ─── Configuration ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "poi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

# Target cities and their approximate bounding boxes [south, west, north, east]
# We'll use area queries instead for better accuracy
TARGET_CITIES = {
    "Malang": {"area_name": "Kota Malang", "kecamatan": ["Lowokwaru", "Klojen", "Blimbing", "Sukun", "Kedungkandang"]},
    "Surabaya": {"area_name": "Kota Surabaya", "kecamatan": ["Gubeng", "Tegalsari", "Genteng", "Wonokromo", "Rungkut"]},
    "Yogyakarta": {"area_name": "Kota Yogyakarta", "kecamatan": ["Gondokusuman", "Umbulharjo", "Kotagede", "Mergangsan", "Danurejan"]},
    "Bandung": {"area_name": "Kota Bandung", "kecamatan": ["Coblong", "Bandung Wetan", "Sumur Bandung", "Cicendo", "Lengkong"]},
    "Semarang": {"area_name": "Kota Semarang", "kecamatan": ["Semarang Tengah", "Semarang Selatan", "Candisari", "Gajahmungkur", "Banyumanik"]},
}

# Business types to search for (OSM amenity/shop tags)
BUSINESS_TYPES = {
    "restaurant": {
        "tags": [
            '["amenity"="restaurant"]',
            '["amenity"="fast_food"]',
        ],
        "label": "Restoran",
    },
    "cafe": {
        "tags": [
            '["amenity"="cafe"]',
            '["shop"="coffee"]',
        ],
        "label": "Kafe",
    },
    "food_stall": {
        "tags": [
            '["amenity"="food_court"]',
            '["shop"="bakery"]',
            '["shop"="pastry"]',
        ],
        "label": "Warung/Bakery",
    },
    "laundry": {
        "tags": [
            '["shop"="laundry"]',
            '["shop"="dry_cleaning"]',
        ],
        "label": "Laundry",
    },
    "convenience": {
        "tags": [
            '["shop"="convenience"]',
            '["shop"="supermarket"]',
            '["shop"="minimarket"]',
        ],
        "label": "Minimarket/Toko",
    },
}

# Known franchise brands in Indonesia for franchise_ratio calculation
FRANCHISE_BRANDS = {
    "mixue", "mie gacoan", "sabana", "mcdonald", "mcd", "kfc",
    "pizza hut", "burger king", "starbucks", "j.co", "jco",
    "chatime", "kopi kenangan", "fore coffee", "janji jiwa",
    "indomaret", "alfamart", "alfamidi", "lawson", "familymart",
    "hokben", "hoka hoka bento", "yoshinoya", "marugame",
    "richeese", "wingstop", "domino", "haus", "teguk",
    "kebab turki", "xendit", "solaria", "es teler 77",
}


def query_overpass(overpass_query: str, retries: int = 3) -> dict:
    """Send a query to Overpass API with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": overpass_query},
                timeout=60,
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait_time = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  HTTP {response.status_code}: {response.text[:200]}")
                time.sleep(10)
        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt + 1}")
            time.sleep(15)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(10)
    return {"elements": []}


def count_pois_in_city(city_name: str, city_info: dict, biz_type: str, biz_config: dict) -> list:
    """Count POIs for a business type across kecamatan in a city."""
    results = []
    area_name = city_info["area_name"]

    for tag in biz_config["tags"]:
        # Build Overpass query for the whole city
        query = f"""
[out:json][timeout:45];
area["name"="{area_name}"]["admin_level"]->.city;
(
  nwr{tag}(area.city);
);
out center tags;
"""
        print(f"  Querying {city_name} for {tag}...")
        data = query_overpass(query)
        elements = data.get("elements", [])
        print(f"    Found {len(elements)} POIs")

        for element in elements:
            tags = element.get("tags", {})
            name = tags.get("name", "").strip()

            # Get coordinates
            lat = element.get("lat") or element.get("center", {}).get("lat")
            lon = element.get("lon") or element.get("center", {}).get("lon")

            # Check if franchise
            is_franchise = False
            if name:
                name_lower = name.lower()
                for brand in FRANCHISE_BRANDS:
                    if brand in name_lower:
                        is_franchise = True
                        break

            results.append({
                "city": city_name,
                "business_type": biz_type,
                "business_label": biz_config["label"],
                "osm_id": element.get("id", ""),
                "osm_type": element.get("type", ""),
                "name": name,
                "amenity": tags.get("amenity", tags.get("shop", "")),
                "cuisine": tags.get("cuisine", ""),
                "lat": lat,
                "lon": lon,
                "is_franchise": is_franchise,
                "brand": tags.get("brand", ""),
                "phone": tags.get("phone", tags.get("contact:phone", "")),
                "website": tags.get("website", ""),
                "opening_hours": tags.get("opening_hours", ""),
            })

        # Be polite — wait between queries
        time.sleep(2)

    return results


def main():
    """Main collection loop."""
    print("=" * 70)
    print(" Overpass API POI Collection — LokaSense / Pasarint")
    print("=" * 70)

    all_results = []

    for city_name, city_info in TARGET_CITIES.items():
        print(f"\n{'─' * 50}")
        print(f"  City: {city_name}")
        print(f"{'─' * 50}")

        for biz_type, biz_config in BUSINESS_TYPES.items():
            results = count_pois_in_city(city_name, city_info, biz_type, biz_config)
            all_results.extend(results)
            print(f"    {biz_config['label']}: {len(results)} POIs")
            time.sleep(3)  # polite delay between business types

    # Save all results
    output_file = OUTPUT_DIR / "overpass_poi.csv"
    if all_results:
        fieldnames = all_results[0].keys()
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\n{'=' * 70}")
    print(f" DONE: {len(all_results)} total POIs saved to {output_file}")
    print(f"{'=' * 70}")

    # Summary statistics
    city_counts = Counter(r["city"] for r in all_results)
    biz_counts = Counter(r["business_type"] for r in all_results)
    franchise_count = sum(1 for r in all_results if r["is_franchise"])

    print(f"\n  POIs per city:")
    for city, count in city_counts.most_common():
        print(f"    {city:20s} {count:>5}")

    print(f"\n  POIs per business type:")
    for biz, count in biz_counts.most_common():
        print(f"    {biz:20s} {count:>5}")

    print(f"\n  Franchise POIs: {franchise_count} / {len(all_results)} ({franchise_count / max(len(all_results), 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()

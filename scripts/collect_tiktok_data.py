#!/usr/bin/env python3
"""
Collect TikTok data (captions, hashtags, engagement) via Apify free tier.
Uses Apify's official client to run a public TikTok scraper actor.

Requires:
    - APIFY_API_TOKEN in .env file (Register free at apify.com for $5/mo credits)
    - pip install apify-client

Usage:
    python scripts/collect_tiktok_data.py
"""
import os
import csv
import time
from pathlib import Path
from dotenv import load_dotenv

# Try to import apify_client gracefully
try:
    from apify_client import ApifyClient
except ImportError:
    print("⚠ apify-client not installed. Please run: pip install apify-client")
    exit(1)

# ─── Configuration ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
ENV_DIR = BASE_DIR.parent
load_dotenv(ENV_DIR / ".env")

OUTPUT_DIR = BASE_DIR / "data" / "social_media"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Apify Token
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN", "")

# Search queries (matching our target areas)
SEARCH_QUERIES = [
    {"query": "ayam geprek lowokwaru malang", "city": "Malang", "kecamatan": "Lowokwaru", "biz": "ayam geprek"},
    {"query": "cafe hits klojen malang", "city": "Malang", "kecamatan": "Klojen", "biz": "cafe"},
    {"query": "kuliner gubeng surabaya", "city": "Surabaya", "kecamatan": "Gubeng", "biz": "kuliner general"},
    {"query": "tempat makan umbulharjo jogja", "city": "Yogyakarta", "kecamatan": "Umbulharjo", "biz": "kedai makan"},
    {"query": "jajanan bandung wetan", "city": "Bandung", "kecamatan": "Bandung Wetan", "biz": "jajanan"},
]

# Using a popular, cost-effective TikTok scraper actor on Apify
ACTOR_ID = "tehotnysloup/tiktok-scraper" # Replace with 'clockwork/tiktok-scraper' if preferred


def scrape_tiktok_apify(query: str, max_items: int = 50) -> list:
    """Run Apify TikTok Scraper for a specific search query."""
    client = ApifyClient(APIFY_TOKEN)
    
    # Prepare the Actor input
    run_input = {
        "searchQueries": [query],
        "resultsPerPage": max_items,
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
        "shouldDownloadSubtitles": False,
    }

    try:
        # Run the Actor and wait for it to finish
        print(f"    Running actor {ACTOR_ID} on Apify cloud...")
        run = client.actor(ACTOR_ID).call(run_input=run_input, memory_mbytes=1024)

        # Fetch and parse results
        results = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            results.append(item)
            
        return results
    except Exception as e:
        print(f"    ⚠ Apify error: {e}")
        return []


def main():
    print("=" * 70)
    print(" TikTok Data Collection (via Apify) — LokaSense / Pasarint")
    print("=" * 70)

    if not APIFY_TOKEN:
        print("\n⚠ APIFY_API_TOKEN not found in .env")
        print("  Please register for a free account at https://apify.com/")
        print("  Generate a personal API token in Settings -> Integrations")
        print("  Add it to your .env file: APIFY_API_TOKEN=apify_api_...")
        return

    all_posts = []
    
    for i, sq in enumerate(SEARCH_QUERIES):
        query = sq["query"]
        print(f"\n[{i+1}/{len(SEARCH_QUERIES)}] Searching TikTok for: '{query}'")
        
        items = scrape_tiktok_apify(query, max_items=30)
        print(f"    Got {len(items)} posts.")
        
        for item in items:
            text = item.get("text", "")
            if not text:
                continue
                
            author = item.get("authorMeta", {}).get("name", "")
            create_time = item.get("createTimeISO", "")
            digg_count = item.get("diggCount", 0)  # likes
            share_count = item.get("shareCount", 0)
            comment_count = item.get("commentCount", 0)
            play_count = item.get("playCount", 0)
            is_ad = item.get("isAd", False)
            
            all_posts.append({
                "text": text.strip().replace("\n", " "),
                "source": "tiktok",
                "timestamp": create_time,
                "area_hint": sq["kecamatan"],
                "city": sq["city"],
                "business_hint": sq["biz"],
                "author": author,
                "likes": digg_count,
                "shares": share_count,
                "comments": comment_count,
                "views": play_count,
                "is_ad": is_ad
            })
            
        time.sleep(2)  # Short pause between actor runs
        
    output_file = OUTPUT_DIR / "tiktok_data.csv"
    if all_posts:
        fieldnames = all_posts[0].keys()
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_posts)
            
    print(f"\n{'=' * 70}")
    print(f" DONE: {len(all_posts)} TikTok posts saved to {output_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

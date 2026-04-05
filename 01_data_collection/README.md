# 01 — Data Collection

This module handles all data acquisition for LokaSense.

## Scripts

| Script | Source | Output | API Key Required |
|--------|--------|--------|-----------------|
| `collect_gmaps_reviews.py` | Google Maps Places API (New) | `data/social_media/gmaps_reviews.csv` | `GOOGLE_MAP` in `.env` |
| `collect_overpass_poi.py` | OpenStreetMap Overpass API | `data/poi/overpass_poi.csv` | None (free) |
| `collect_tiktok_data.py` | Apify TikTok Scraper | `data/social_media/tiktok_data.csv` | `APIFY_API_TOKEN` in `.env` |

## Execution Order
```bash
# 1. Run dataset download first (HuggingFace + GeoBoundaries)
python 01_data_collection/setup_datasets.py

# 2. Collect POI data (free, no API key)
python 01_data_collection/collect_overpass_poi.py

# 3. Collect Google Maps reviews (requires GOOGLE_MAP key)
python 01_data_collection/collect_gmaps_reviews.py

# 4. Optional: Collect TikTok data (requires APIFY_API_TOKEN)
python 01_data_collection/collect_tiktok_data.py
```

## Output Data Schema

### gmaps_reviews.csv
| Column | Type | Description |
|--------|------|-------------|
| text | str | Review text |
| source | str | Always "google_maps" |
| timestamp | str | ISO8601 publish time |
| area_hint | str | Kecamatan name |
| city | str | City name |
| business_hint | str | Business category |
| place_name | str | Business name |
| review_rating | int | 1-5 stars |
| lat/lng | float | Coordinates |

### overpass_poi.csv
| Column | Type | Description |
|--------|------|-------------|
| city | str | City name |
| business_type | str | Category key |
| name | str | Business name |
| is_franchise | bool | Franchise brand detected |
| lat/lon | float | Coordinates |

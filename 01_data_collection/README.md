# 01 — Data Collection

This module now uses a Scrapling-first public scraping pipeline for in-domain bootstrap data.

## Scripts

| Script | Source | Output | Notes |
|--------|--------|--------|-------|
| `collect_social_bootstrap.py` | TikTok, Instagram, X, Google Maps | `data/scraped/*` + platform CSVs | Main entrypoint |
| `collect_tiktok_data.py` | TikTok only | `data/social_media/tiktok_data.csv` | Uses the same Scrapling pipeline |
| `collect_gmaps_reviews.py` | Google Maps Places API (New) | `data/social_media/gmaps_reviews.csv` | Requires `GOOGLE_MAP` |
| `collect_overpass_poi.py` | OpenStreetMap Overpass API | `data/poi/overpass_poi.csv` | Free |
| `setup_datasets.py` | Geo assets + optional legacy corpora | `data/geojson/*` | SmSA / NERP are no longer required |

## Install

```bash
pip install "scrapling[fetchers]"
scrapling install
```

Playwright control is used through Scrapling `DynamicSession.page_action`, so the browser install step is required for TikTok / Instagram / X extraction.

## Recommended Flow

```bash
# 1. Prepare geo assets
python 01_data_collection/setup_datasets.py

# 2. Collect POI names for place matching
python 01_data_collection/collect_overpass_poi.py

# 3. Collect or refresh Google Maps reviews when quota allows
python 01_data_collection/collect_gmaps_reviews.py

# 4. Build scraped social bootstrap datasets
python 01_data_collection/collect_social_bootstrap.py --headless
```

## Outputs

### `data/social_media/tiktok_data.csv`

Primary TikTok caption corpus with normalized text, raw caption, engagement metadata, query provenance, and scrape mode.

### `data/scraped/signal_bootstrap.csv`

Merged in-domain text corpus built from TikTok, Instagram, X, and Google Maps. This replaces the old dependency on missing `smsa` downloads for the active weak-labeling path.

### `data/scraped/ner_bootstrap.jsonl`

Short-form public text with tokenization, gazetteer-driven candidate spans, and weak `LOC` / `ORG` tags. This replaces the old dependency on missing `nerp` files for domain-adaptation bootstrap.

### `data/scraped/manifest.json`

Collector provenance, row counts, dedupe totals, blocked-source reasons, and bootstrap artifact counts.

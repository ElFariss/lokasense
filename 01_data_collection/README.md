# 01 — Data Collection

This module is now public-first and free-by-default.

The maintained path is:
1. collect free geospatial metadata from OpenStreetMap Overpass,
2. scrape public social text with Scrapling plus Playwright control,
3. rebuild `signal_bootstrap.csv` and `ner_bootstrap.jsonl` from those public sources.

Google Maps is no longer part of the default bootstrap flow. It is an optional enrichment path and requires an explicit billable confirmation flag.

## Scripts

| Script | Source | Output | Notes |
|--------|--------|--------|-------|
| `collect_social_bootstrap.py` | TikTok, Instagram, X | `data/scraped/*` + platform CSVs | Main entrypoint, public scraping only by default |
| `collect_tiktok_data.py` | TikTok only | `data/social_media/tiktok_data.csv` | Uses the same Scrapling pipeline |
| `collect_overpass_poi.py` | OpenStreetMap Overpass API | `data/poi/overpass_poi.csv` | Free, used for place matching and franchise hints |
| `collect_gmaps_reviews.py` | Google Maps Places API (New) | `data/social_media/gmaps_reviews.csv` | Optional, billable, requires `--confirm-billable` |
| `setup_datasets.py` | Geo assets + optional legacy corpora | `data/geojson/*` | SmSA and NERP are no longer required |

## Install

```bash
pip install "scrapling[fetchers]"
scrapling install
```

Scrapling uses Playwright control through `DynamicSession.page_action`, so the browser install step is required for TikTok, Instagram, and X extraction.

## Recommended Free Workflow

```bash
# 1. Prepare geo assets
python 01_data_collection/setup_datasets.py

# 2. Collect POI names for place matching
python 01_data_collection/collect_overpass_poi.py

# 3. Build public social bootstrap datasets
python 01_data_collection/collect_social_bootstrap.py --headless
```

## Optional Google Maps Enrichment

Only run this if you intentionally want the paid API path:

```bash
python 01_data_collection/collect_gmaps_reviews.py --confirm-billable
python 01_data_collection/collect_social_bootstrap.py --include-gmaps-cache --headless
```

If you want the social bootstrap entrypoint itself to refresh Google Maps, it also requires explicit confirmation:

```bash
python 01_data_collection/collect_social_bootstrap.py --refresh-gmaps --confirm-billable --include-gmaps-cache --headless
```

## Outputs

### `data/social_media/tiktok_data.csv`

Primary TikTok caption corpus with normalized text, raw caption, engagement metadata, query provenance, and scrape mode.

### `data/scraped/signal_bootstrap.csv`

Merged in-domain text corpus built from public scraped social sources by default. This replaces the old dependency on missing `smsa` downloads for the active weak-labeling path.

### `data/scraped/ner_bootstrap.jsonl`

Short-form public text with tokenization, gazetteer-driven candidate spans, and weak `LOC` / `ORG` tags. This replaces the old dependency on missing `nerp` files for domain-adaptation bootstrap.

### `data/scraped/manifest.json`

Collector provenance, row counts, blocked-source reasons, dedupe totals, and bootstrap artifact counts. It now records whether Google Maps was included and whether a billable refresh was used.

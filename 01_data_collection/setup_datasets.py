#!/usr/bin/env python3
"""
01_data_collection/setup_datasets.py
Prepare non-scraped assets that the pipeline still needs.

Default behavior now treats missing packaged corpora like SmSA / NERP as optional.
The active bootstrap path is:
    python 01_data_collection/collect_social_bootstrap.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def verify_existing() -> None:
    print("\n" + "=" * 60)
    print(" Verifying Existing Datasets")
    print("=" * 60)

    checks = [
        ("IndoLEM NER (UGM)", DATA_DIR / "indolem_ner" / "indolem" / "ner" / "data" / "nerugm"),
        ("IndoLEM NER (UI)", DATA_DIR / "indolem_ner" / "indolem" / "ner" / "data" / "nerui"),
        ("NusaX Sentiment", DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian"),
        ("Admin Gazetteer", DATA_DIR / "geospatial" / "Wilayah-Administratif-Indonesia" / "csv"),
        ("Overpass POI", DATA_DIR / "poi" / "overpass_poi.csv"),
        ("Social Bootstrap Manifest", DATA_DIR / "scraped" / "manifest.json"),
    ]

    for name, path in checks:
        if path.exists():
            if path.is_dir():
                print(f"  OK {name}: {len(list(path.glob('*')))} files")
            else:
                print(f"  OK {name}: present")
        else:
            print(f"  Missing {name}: {path}")


def download_indonesian_sentiment() -> None:
    target_dir = DATA_DIR / "huggingface" / "indonesian_sentiment"
    if (target_dir / "train.csv").exists():
        print("Indonesian Sentiment already downloaded")
        return

    print("Downloading Indonesian Sentiment (legacy optional corpus)...")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset

        dataset = load_dataset("sepidmnorozy/Indonesian_sentiment")
        for split_name, split in dataset.items():
            output_file = target_dir / f"{split_name}.csv"
            split.to_csv(str(output_file), index=False)
            print(f"  Saved {split_name}: {len(split)} samples")
    except Exception as exc:
        print(f"  Could not download Indonesian Sentiment: {exc}")


def download_geoboundaries() -> None:
    import requests

    geojson_dir = DATA_DIR / "geojson"
    geojson_dir.mkdir(parents=True, exist_ok=True)
    urls = {
        "adm1": "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM1/",
        "adm2": "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM2/",
    }

    for level, api_url in urls.items():
        geojson_file = geojson_dir / f"indonesia_{level}.geojson"
        if geojson_file.exists():
            print(f"GeoBoundaries {level.upper()} already downloaded")
            continue
        print(f"Downloading GeoBoundaries Indonesia {level.upper()}...")
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            download_url = response.json().get("gjDownloadURL", "")
            if not download_url:
                print(f"  Missing download URL for {level}")
                continue
            geo_response = requests.get(download_url, timeout=120)
            geo_response.raise_for_status()
            with open(geojson_file, "w", encoding="utf-8") as handle:
                handle.write(geo_response.text)
            feature_count = len(json.loads(geo_response.text).get("features", []))
            print(f"  Saved {geojson_file.name} ({feature_count} features)")
        except Exception as exc:
            print(f"  Error downloading {level}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare non-scraped assets for LokaSense.")
    parser.add_argument(
        "--include-legacy-hf",
        action="store_true",
        help="Also download the legacy Indonesian Sentiment corpus. SmSA and NERP stay replaced by scraped bootstrap data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(" LokaSense Dataset Setup")
    print("=" * 60)
    verify_existing()
    print("\n" + "=" * 60)
    print(" Core Assets")
    print("=" * 60)
    download_geoboundaries()
    if args.include_legacy_hf:
        print("\n" + "=" * 60)
        print(" Optional Legacy Corpus")
        print("=" * 60)
        download_indonesian_sentiment()
    else:
        print("\nSmSA and NERP are no longer required. Build scraped replacements with:")
        print("  python 01_data_collection/collect_social_bootstrap.py")
    print("\nSetup complete.")

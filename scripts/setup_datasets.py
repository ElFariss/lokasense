#!/usr/bin/env python3
"""
Download datasets from HuggingFace that are not yet present locally.
Downloads: SmSA (from IndoNLU), Indonesian Sentiment, GeoBoundaries GeoJSON.

Usage:
    python scripts/setup_datasets.py
"""
import os
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def download_huggingface_datasets():
    """Download HuggingFace datasets using the `datasets` library."""
    from datasets import load_dataset

    # ─── 1. SmSA (Sentiment Analysis from IndoNLU) ──────────────────────────
    smsa_dir = DATA_DIR / "huggingface" / "smsa"
    if not (smsa_dir / "train.csv").exists():
        print("📥 Downloading SmSA (IndoNLU sentiment)...")
        smsa_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset("indonlp/indonlu", "smsa")
        for split_name in ["train", "validation", "test"]:
            if split_name in ds:
                split = ds[split_name]
                output_file = smsa_dir / f"{split_name}.csv"
                split.to_csv(str(output_file), index=False)
                print(f"  ✅ {split_name}: {len(split)} samples → {output_file}")
    else:
        print("✅ SmSA already downloaded")

    # ─── 2. Indonesian Sentiment ────────────────────────────────────────────
    indosent_dir = DATA_DIR / "huggingface" / "indonesian_sentiment"
    if not (indosent_dir / "train.csv").exists():
        print("📥 Downloading Indonesian Sentiment...")
        indosent_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset("sepidmnorozy/Indonesian_sentiment")
        for split_name in ds.keys():
            split = ds[split_name]
            output_file = indosent_dir / f"{split_name}.csv"
            split.to_csv(str(output_file), index=False)
            print(f"  ✅ {split_name}: {len(split)} samples → {output_file}")
    else:
        print("✅ Indonesian Sentiment already downloaded")

    # ─── 3. IndoNLU NERP (for FNB entity) ──────────────────────────────────
    nerp_dir = DATA_DIR / "huggingface" / "nerp"
    if not (nerp_dir / "train.csv").exists():
        print("📥 Downloading IndoNLU NERP...")
        nerp_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset("indonlp/indonlu", "nerp")
        for split_name in ds.keys():
            split = ds[split_name]
            output_file = nerp_dir / f"{split_name}.csv"
            split.to_csv(str(output_file), index=False)
            print(f"  ✅ {split_name}: {len(split)} samples → {output_file}")
    else:
        print("✅ IndoNLU NERP already downloaded")


def download_geoboundaries():
    """Download GeoBoundaries Indonesia GeoJSON (ADM2 level)."""
    import requests

    geojson_dir = DATA_DIR / "geojson"
    geojson_dir.mkdir(parents=True, exist_ok=True)

    # GeoBoundaries API endpoint for Indonesia ADM2
    # This fetches the simplified GeoJSON for kabupaten/kota level
    urls = {
        "adm1": "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM1/",
        "adm2": "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM2/",
    }

    for level, api_url in urls.items():
        geojson_file = geojson_dir / f"indonesia_{level}.geojson"
        if geojson_file.exists():
            print(f"✅ GeoBoundaries {level.upper()} already downloaded")
            continue

        print(f"📥 Downloading GeoBoundaries Indonesia {level.upper()}...")
        try:
            # First, get the download URL from the API
            response = requests.get(api_url, timeout=30)
            if response.status_code == 200:
                metadata = response.json()
                download_url = metadata.get("gjDownloadURL", "")

                if download_url:
                    print(f"  Fetching GeoJSON from: {download_url[:80]}...")
                    geo_response = requests.get(download_url, timeout=120)
                    if geo_response.status_code == 200:
                        with open(geojson_file, "w", encoding="utf-8") as f:
                            f.write(geo_response.text)
                        print(f"  ✅ Saved to {geojson_file}")

                        # Quick stats
                        geojson_data = json.loads(geo_response.text)
                        n_features = len(geojson_data.get("features", []))
                        print(f"  📊 {n_features} features")
                    else:
                        print(f"  ⚠ Download failed: HTTP {geo_response.status_code}")
                else:
                    print(f"  ⚠ No download URL in API response")
            else:
                print(f"  ⚠ API error: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ⚠ Error downloading {level}: {e}")
            print(f"  💡 Try manual download from: https://data.humdata.org/dataset/geoboundaries-admin-boundaries-for-indonesia")


def verify_existing_datasets():
    """Verify all previously downloaded datasets are intact."""
    print("\n" + "=" * 60)
    print(" Verifying Existing Datasets")
    print("=" * 60)

    checks = [
        ("IndoLEM NER (UGM)", DATA_DIR / "indolem_ner" / "indolem" / "ner" / "data" / "nerugm"),
        ("IndoLEM NER (UI)", DATA_DIR / "indolem_ner" / "indolem" / "ner" / "data" / "nerui"),
        ("NusaX Sentiment (ID)", DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian"),
        ("Admin Gazetteer", DATA_DIR / "geospatial" / "Wilayah-Administratif-Indonesia" / "csv"),
    ]

    all_ok = True
    for name, path in checks:
        if path.exists():
            if path.is_dir():
                n_files = len(list(path.glob("*")))
                print(f"  ✅ {name}: {n_files} files in {path.name}/")
            else:
                size_kb = path.stat().st_size / 1024
                print(f"  ✅ {name}: {size_kb:.0f} KB")
        else:
            print(f"  ❌ {name}: NOT FOUND at {path}")
            all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print(" LokaSense Dataset Setup")
    print("=" * 60)

    # 1. Verify existing datasets
    verify_existing_datasets()

    # 2. Download new datasets
    print("\n" + "=" * 60)
    print(" Downloading New Datasets")
    print("=" * 60)
    download_huggingface_datasets()

    # 3. Download geoboundaries
    print("\n" + "=" * 60)
    print(" Downloading GeoBoundaries")
    print("=" * 60)
    download_geoboundaries()

    print("\n" + "=" * 60)
    print(" ✅ Setup Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

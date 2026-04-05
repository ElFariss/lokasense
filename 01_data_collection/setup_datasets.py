#!/usr/bin/env python3
"""
01_data_collection/setup_datasets.py
Download datasets from HuggingFace + GeoBoundaries. Run this FIRST.

Usage:
    python 01_data_collection/setup_datasets.py
"""
import os
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def download_huggingface_datasets():
    """Download HuggingFace datasets."""
    import requests

    # ---  SmSA from IndoNLU (download raw parquet/csv directly) ---
    smsa_dir = DATA_DIR / "huggingface" / "smsa"
    if not (smsa_dir / "train.csv").exists():
        print("📥 Downloading SmSA (IndoNLU sentiment)...")
        smsa_dir.mkdir(parents=True, exist_ok=True)
        
        # IndoNLU SmSA was upstreamed to parquet; download from raw data on HF
        base_url = "https://huggingface.co/datasets/indonlp/indonlu/resolve/main/smsa"
        for split in ["train", "valid", "test"]:
            url = f"{base_url}/{split}_preprocess.csv"
            print(f"  Trying {url}...")
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    out = smsa_dir / f"{split}.csv"
                    with open(out, "wb") as f:
                        f.write(r.content)
                    lines = r.text.count('\n')
                    print(f"  ✅ {split}: ~{lines} samples")
                else:
                    print(f"  ⚠ {split}: HTTP {r.status_code}")
            except Exception as e:
                print(f"  ⚠ {split}: {e}")

        # If direct CSV didn't work, try another approach:
        if not (smsa_dir / "train.csv").exists():
            print("  Trying alternative download via HuggingFace API...")
            try:
                from huggingface_hub import hf_hub_download
                for split in ["train", "valid", "test"]:
                    filepath = hf_hub_download(
                        repo_id="indonlp/indonlu",
                        filename=f"smsa/{split}_preprocess.csv",
                        repo_type="dataset"
                    )
                    import shutil
                    shutil.copy(filepath, smsa_dir / f"{split}.csv")
                    print(f"  ✅ {split} downloaded via hub")
            except Exception as e:
                print(f"  ⚠ Hub download failed: {e}")
    else:
        print("✅ SmSA already downloaded")

    # --- Indonesian Sentiment ---
    indosent_dir = DATA_DIR / "huggingface" / "indonesian_sentiment"
    if not (indosent_dir / "train.csv").exists():
        print("📥 Downloading Indonesian Sentiment...")
        indosent_dir.mkdir(parents=True, exist_ok=True)
        try:
            from datasets import load_dataset
            ds = load_dataset("sepidmnorozy/Indonesian_sentiment")
            for split_name in ds.keys():
                split = ds[split_name]
                output_file = indosent_dir / f"{split_name}.csv"
                split.to_csv(str(output_file), index=False)
                print(f"  ✅ {split_name}: {len(split)} samples")
        except Exception as e:
            print(f"  ⚠ Could not download: {e}")
    else:
        print("✅ Indonesian Sentiment already downloaded")

    # --- IndoNLU NERP ---
    nerp_dir = DATA_DIR / "huggingface" / "nerp"
    if not (nerp_dir / "train.csv").exists():
        print("📥 Downloading IndoNLU NERP...")
        nerp_dir.mkdir(parents=True, exist_ok=True)
        base_url = "https://huggingface.co/datasets/indonlp/indonlu/resolve/main/nerp"
        for split in ["train", "valid", "test"]:
            url = f"{base_url}/{split}_preprocess.txt"
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    out = nerp_dir / f"{split}.txt"
                    with open(out, "wb") as f:
                        f.write(r.content)
                    print(f"  ✅ {split}: downloaded")
                else:
                    print(f"  ⚠ {split}: HTTP {r.status_code}")
            except Exception as e:
                print(f"  ⚠ {split}: {e}")
    else:
        print("✅ IndoNLU NERP already downloaded")


def download_geoboundaries():
    """Download GeoBoundaries Indonesia GeoJSON (ADM2 level)."""
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
            print(f"✅ GeoBoundaries {level.upper()} already downloaded")
            continue

        print(f"📥 Downloading GeoBoundaries Indonesia {level.upper()}...")
        try:
            response = requests.get(api_url, timeout=30)
            if response.status_code == 200:
                metadata = response.json()
                download_url = metadata.get("gjDownloadURL", "")
                if download_url:
                    print(f"  Fetching from: {download_url[:80]}...")
                    geo_response = requests.get(download_url, timeout=120)
                    if geo_response.status_code == 200:
                        with open(geojson_file, "w", encoding="utf-8") as f:
                            f.write(geo_response.text)
                        geojson_data = json.loads(geo_response.text)
                        n_features = len(geojson_data.get("features", []))
                        print(f"  ✅ Saved ({n_features} features)")
        except Exception as e:
            print(f"  ⚠ Error: {e}")


def verify_existing():
    """Quick verify of already-downloaded datasets."""
    print("\n" + "=" * 60)
    print(" Verifying Existing Datasets")
    print("=" * 60)

    checks = [
        ("IndoLEM NER (UGM)", DATA_DIR / "indolem_ner" / "indolem" / "ner" / "data" / "nerugm"),
        ("IndoLEM NER (UI)",  DATA_DIR / "indolem_ner" / "indolem" / "ner" / "data" / "nerui"),
        ("NusaX Sentiment",   DATA_DIR / "nusax_sentiment" / "nusax" / "datasets" / "sentiment" / "indonesian"),
        ("Admin Gazetteer",   DATA_DIR / "geospatial" / "Wilayah-Administratif-Indonesia" / "csv"),
    ]

    for name, path in checks:
        if path.exists():
            if path.is_dir():
                n = len(list(path.glob("*")))
                print(f"  ✅ {name}: {n} files")
            else:
                print(f"  ✅ {name}: {path.stat().st_size / 1024:.0f} KB")
        else:
            print(f"  ❌ {name}: NOT FOUND at {path}")


if __name__ == "__main__":
    print("=" * 60)
    print(" LokaSense Dataset Setup")
    print("=" * 60)
    verify_existing()
    print("\n" + "=" * 60)
    print(" Downloading New Datasets")
    print("=" * 60)
    download_huggingface_datasets()
    print("\n" + "=" * 60)
    print(" Downloading GeoBoundaries")
    print("=" * 60)
    download_geoboundaries()
    print("\n✅ Setup Complete!")

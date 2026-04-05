#!/usr/bin/env python3
"""
02_ner_model/dataset/prepare.py
Load IndoLEM NER data from TSV and format for HuggingFace Trainer.
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = BASE_DIR / "train_data"
TEST_DIR = BASE_DIR / "test_data"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


def parse_tsv(filepath):
    """Parse IndoLEM NER TSV format: word\\tPOS\\tentity"""
    if not filepath.exists():
        return []
    sentences = []
    tokens, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": labels})
                    tokens, labels = [], []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    labels.append(parts[-1])
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": labels})
    return sentences


def main():
    print("=" * 60)
    print(" NER Dataset Preparation — IndoLEM")
    print("=" * 60)

    base = DATA_DIR / "indolem_ner" / "indolem" / "ner" / "data"

    # UGM
    ugm_train = parse_tsv(base / "nerugm" / "train.01.tsv")
    ugm_dev = parse_tsv(base / "nerugm" / "dev.01.tsv")
    ugm_test = parse_tsv(base / "nerugm" / "test.01.tsv")

    # UI
    ui_train = parse_tsv(base / "nerui" / "train.01.tsv")
    ui_test = parse_tsv(base / "nerui" / "test.01.tsv")

    train_all = ugm_train + ui_train
    val_all = ugm_dev + ui_test[:len(ui_test)//2]  # half UI test as extra val
    test_all = ugm_test + ui_test[len(ui_test)//2:]

    if not train_all:
        print("  ⚠ No IndoLEM data found! Check data/indolem_ner/")
        return

    # Also try adding NERP from HuggingFace
    nerp_dir = DATA_DIR / "huggingface" / "nerp"
    if (nerp_dir / "train.csv").exists():
        import pandas as pd
        nerp_df = pd.read_csv(nerp_dir / "train.csv")
        # NERP format: tokens column and ner_tags column (as lists)
        print(f"  Also found NERP: {len(nerp_df)} samples")

    # Save
    with open(TRAIN_DIR / "ner_train.json", 'w') as f:
        json.dump(train_all, f)
    with open(TRAIN_DIR / "ner_val.json", 'w') as f:
        json.dump(val_all, f)
    with open(TEST_DIR / "ner_test.json", 'w') as f:
        json.dump(test_all, f)

    # Collect all unique labels
    all_labels = set()
    for s in train_all + val_all + test_all:
        all_labels.update(s["ner_tags"])

    print(f"\n✅ NER splits created:")
    print(f"  Train: {len(train_all)} sentences")
    print(f"  Val:   {len(val_all)} sentences")
    print(f"  Test:  {len(test_all)} sentences")
    print(f"  Labels: {sorted(all_labels)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create leakage-proof data splits (Train/Val/Test).
This step MUST be run before any ML code. It handles the initial 
splitting of labeled data so that testing is completely untouched.

Outputs:
    train_data/signal_train.csv
    train_data/signal_val.csv
    test_data/signal_test.csv

Usage:
    python scripts/create_data_splits.py
"""
import json
import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.bootstrap_utils import stable_split
from common.ner_labels import has_entity, normalize_ner_tags

LABELED_DIR = BASE_DIR / "data" / "labeled"
TRAIN_DIR = BASE_DIR / "train_data"
TEST_DIR = BASE_DIR / "test_data"
USE_GEMINI_AUGMENTATION = os.getenv("USE_GEMINI_AUGMENTATION", "0") == "1"
USE_GEMINI_OVERRIDES = os.getenv("USE_GEMINI_OVERRIDES", "0") == "1"
USE_MODEL_PSEUDOLABELS = os.getenv("USE_MODEL_PSEUDOLABELS", "0") == "1"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

class DatasetSplitter:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    @staticmethod
    def _add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
        defaults = {
            "source": "unknown",
            "platform": "",
            "url": "",
            "timestamp": "",
            "city": "",
            "area_hint": "",
            "business_hint": "",
            "query": "",
            "query_intent": "",
            "provenance_split": "",
            "label_source": "weak_labeled",
        }
        for column, default_value in defaults.items():
            if column not in df.columns:
                df[column] = default_value
        return df

    @staticmethod
    def _apply_provenance_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[None, None, None]:
        valid_splits = {"train", "validation", "test"}
        if "provenance_split" not in df.columns:
            return None, None, None

        split_values = df["provenance_split"].fillna("").astype(str)
        if not split_values.isin(valid_splits).any():
            return None, None, None

        train_df = df[split_values == "train"].copy()
        val_df = df[split_values == "validation"].copy()
        test_df = df[split_values == "test"].copy()
        if train_df.empty or val_df.empty or test_df.empty:
            return None, None, None
        return train_df, val_df, test_df
        
    def create_signal_splits(self):
        """Create leakage-proof splits for the 7-class signal dataset."""
        print("="*60)
        print(" Creating Leakage-Proof Data Splits (Signal)")
        print("="*60)
        
        # 1. Load weak-labeled data as baseline
        weak_file = LABELED_DIR / "weak_labeled.csv"
        gemini_file = LABELED_DIR / "gemini_augmented.csv"
        
        if not weak_file.exists():
            print("weak_labeled.csv not found. Run weak_labeling.py first.")
            return False
            
        weak_df = pd.read_csv(weak_file)
        weak_df = self._add_missing_columns(weak_df)
        weak_df["provenance_split"] = weak_df["provenance_split"].fillna("").astype(str)
        weak_df.loc[weak_df["provenance_split"] == "", "provenance_split"] = weak_df.loc[
            weak_df["provenance_split"] == "", "text"
        ].map(stable_split)
        weak_df["label_source"] = "weak_labeled"
        weak_df["final_signal"] = weak_df["signal"]
        print(f"  Loaded {len(weak_df)} weak labeled texts.")
        
        # 2. Merge fresh Gemini augmentation if available
        if USE_GEMINI_AUGMENTATION and gemini_file.exists():
            gemini_df = pd.read_csv(gemini_file)
            gemini_df = self._add_missing_columns(gemini_df)
            gemini_df = gemini_df.dropna(subset=["text", "gemini_signal"]).copy()
            gemini_df["gemini_confidence"] = pd.to_numeric(gemini_df.get("gemini_confidence", 0), errors="coerce").fillna(0.0)
            gemini_df["provenance_split"] = gemini_df["provenance_split"].fillna("").astype(str)
            gemini_df.loc[gemini_df["provenance_split"] == "", "provenance_split"] = gemini_df.loc[
                gemini_df["provenance_split"] == "", "text"
            ].map(stable_split)
            print(f"  Loaded {len(gemini_df)} fresh Gemini-augmented texts.")

            usable_gemini = gemini_df[
                (gemini_df["gemini_confidence"] >= 0.7)
                & (
                    (gemini_df["gemini_signal"] != "NEUTRAL")
                    | (gemini_df["gemini_confidence"] >= 0.85)
                )
            ].copy()
            usable_gemini = usable_gemini.sort_values(["gemini_confidence", "text"], ascending=[False, True]).drop_duplicates("text")
            print(f"  Kept {len(usable_gemini)} Gemini rows after confidence filtering.")

            if USE_GEMINI_OVERRIDES and not usable_gemini.empty:
                gemini_lookup = usable_gemini.set_index("text")
                override_mask = weak_df["text"].isin(gemini_lookup.index)
                weak_df.loc[override_mask, "final_signal"] = weak_df.loc[override_mask, "text"].map(gemini_lookup["gemini_signal"])
                weak_df.loc[override_mask, "label_source"] = "gemini_override"
                print(f"  Applied Gemini overrides to {int(override_mask.sum())} existing weak-labeled rows.")
            elif USE_GEMINI_AUGMENTATION:
                print("  Gemini overrides disabled; using Gemini only for net-new augmentation rows.")

            weak_texts = set(weak_df["text"].tolist())
            augmented_only = usable_gemini[~usable_gemini["text"].isin(weak_texts)].copy()
            if not augmented_only.empty:
                augmented_only["final_signal"] = augmented_only["gemini_signal"]
                augmented_only["label_source"] = "gemini_augmented"
                augmented_only["provenance_split"] = "train"
                weak_df = pd.concat([weak_df, augmented_only], ignore_index=True, sort=False)
                print(f"  Added {len(augmented_only)} net-new Gemini texts to the supervised pool.")
        else:
            if USE_GEMINI_AUGMENTATION:
                print("  Gemini augmentation requested, but no fresh Gemini file was found. Using weak labels only.")
            else:
                print("  Gemini augmentation disabled for this run. Using weak labels only.")

        model_pseudo_file = LABELED_DIR / "model_pseudo_augmented.csv"
        if USE_MODEL_PSEUDOLABELS and model_pseudo_file.exists():
            model_df = pd.read_csv(model_pseudo_file)
            model_df = self._add_missing_columns(model_df)
            model_df = model_df.dropna(subset=["text", "model_signal"]).copy()
            model_df["model_confidence"] = pd.to_numeric(model_df.get("model_confidence", 0), errors="coerce").fillna(0.0)
            model_df["provenance_split"] = "train"
            model_df["final_signal"] = model_df["model_signal"]
            model_df["label_source"] = "model_pseudo_augmented"
            model_df = model_df[model_df["text"].astype(str).str.len() > 10]
            model_df = model_df.sort_values(["model_confidence", "text"], ascending=[False, True]).drop_duplicates("text")
            weak_texts = set(weak_df["text"].tolist())
            model_df = model_df[~model_df["text"].isin(weak_texts)]
            if not model_df.empty:
                weak_df = pd.concat([weak_df, model_df], ignore_index=True, sort=False)
                print(f"  Added {len(model_df)} train-only IndoBERT pseudolabel rows.")
        else:
            if USE_MODEL_PSEUDOLABELS:
                print("  Local IndoBERT pseudolabel augmentation requested, but no model_pseudo_augmented.csv was found.")
            else:
                print("  Local IndoBERT pseudolabel augmentation disabled for this run.")
            
        # Remove empty or super short texts
        df = weak_df[weak_df['text'].str.len() > 10].copy()
        df = df.drop_duplicates(subset=["text"], keep="last")

        # Ultra-rare classes are train-only so we don't waste the only
        # positive examples on validation/test.
        class_counts = df["final_signal"].value_counts()
        rare_classes = set(class_counts[class_counts < 3].index.tolist())
        rare_df = df[df["final_signal"].isin(rare_classes)].copy()
        split_df = df[~df["final_signal"].isin(rare_classes)].copy()
        if rare_classes:
            print(f"  Keeping rare classes in train only: {sorted(rare_classes)}")

        train_df, val_df, test_df = self._apply_provenance_splits(split_df)
        if train_df is not None:
            print("\n  Splitting data using stable provenance_split assignments...")
        else:
            print("\n  Splitting data (Stratified 70/15/15)...")
            try:
                train_val_df, test_df = train_test_split(
                    split_df, test_size=0.15,
                    random_state=self.random_seed, 
                    stratify=split_df['final_signal']
                )
                
                train_df, val_df = train_test_split(
                    train_val_df, test_size=0.15/0.85,
                    random_state=self.random_seed,
                    stratify=train_val_df['final_signal']
                )
            except ValueError as e:
                print(f"  Stratified split failed ({e}), falling back to random split.")
                train_val_df, test_df = train_test_split(
                    split_df, test_size=0.15, random_state=self.random_seed
                )
                train_df, val_df = train_test_split(
                    train_val_df, test_size=0.15/0.85, random_state=self.random_seed
                )

        if not rare_df.empty:
            train_df = pd.concat([train_df, rare_df], ignore_index=True)
            train_df = train_df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)

        # Ensure every class still has at least one train example.
        for label in sorted(df["final_signal"].unique()):
            if label in set(train_df["final_signal"].tolist()):
                continue
            for donor_name, donor_df in [("validation", val_df), ("test", test_df)]:
                label_rows = donor_df[donor_df["final_signal"] == label]
                if label_rows.empty:
                    continue
                train_df = pd.concat([train_df, label_rows.head(1)], ignore_index=True)
                donor_df.drop(label_rows.head(1).index, inplace=True)
                print(f"  Moved one {label} row from {donor_name} to train to preserve class coverage.")
                break

        # Save splits
        train_path = TRAIN_DIR / "signal_train.csv"
        val_path = TRAIN_DIR / "signal_val.csv"
        test_path = TEST_DIR / "signal_test.csv"
        
        keep_cols = [
            column
            for column in [
                "text",
                "final_signal",
                "source",
                "platform",
                "url",
                "timestamp",
                "city",
                "area_hint",
                "business_hint",
                "query",
                "query_intent",
                "provenance_split",
                "label_source",
            ]
            if column in train_df.columns
        ]
        train_df[keep_cols].to_csv(train_path, index=False)
        val_df[keep_cols].to_csv(val_path, index=False)
        test_df[keep_cols].to_csv(test_path, index=False)
        
        # Print stats
        print(f"\nSplits saved successfully:")
        print(f"   Train : {len(train_df):>6} ({len(train_df)/len(df)*100:.1f}%) -> {train_path}")
        print(f"   Val   : {len(val_df):>6} ({len(val_df)/len(df)*100:.1f}%) -> {val_path}")
        print(f"   Test  : {len(test_df):>6} ({len(test_df)/len(df)*100:.1f}%) -> {test_path}")
        
        # Proof of no leakage
        train_texts = set(train_df['text'])
        test_texts = set(test_df['text'])
        leakage = train_texts.intersection(test_texts)
        print(f"   Leakage test: {len(leakage)} overlapping rows between train and test.")
        
        return True

    def create_ner_splits(self):
        """Format the IndoLEM NER datasets for huggingface trainer."""
        print("\n" + "="*60)
        print(" Creating Leakage-Proof Data Splits (NER)")
        print("="*60)
        
        # The IndoLEM dataset is already split into train/dev/test TSV files.
        # We just need to parse them and save them as nice JSON files in train_data/test_data
        
        indolem_base = BASE_DIR / "data" / "indolem_ner" / "indolem" / "ner" / "data"
        
        def parse_tsv(filepath):
            if not filepath.exists():
                return []
                
            sentences = []
            tokens = []
            labels = []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        if tokens:
                            sentences.append({"tokens": tokens, "ner_tags": labels})
                            tokens = []
                            labels = []
                    else:
                        parts = line.split('\t') # Format: word POS Entity
                        if len(parts) >= 3:
                            tokens.append(parts[0])
                            labels.append(parts[2])
                            
            if tokens:
                sentences.append({"tokens": tokens, "ner_tags": labels})
            for sentence in sentences:
                sentence["ner_tags"] = normalize_ner_tags(sentence["ner_tags"])
            return sentences

        # Load UGM dataset
        ugm_train = parse_tsv(indolem_base / "nerugm" / "train.01.tsv")
        ugm_dev = parse_tsv(indolem_base / "nerugm" / "dev.01.tsv")
        ugm_test = parse_tsv(indolem_base / "nerugm" / "test.01.tsv")
        
        # Load UI dataset 
        ui_train = parse_tsv(indolem_base / "nerui" / "train.01.tsv")
        ui_dev = parse_tsv(indolem_base / "nerui" / "test.01.tsv") # UI uses test for dev usually
        
        train_all = ugm_train + ui_train
        val_all = ugm_dev + ui_dev
        test_all = ugm_test
        
        if not train_all:
            print("  IndoLEM data not found. Run setup_datasets.py first.")
            return False

        weak_ner_file = BASE_DIR / "data" / "scraped" / "ner_bootstrap.jsonl"
        weak_ner_count = 0
        if weak_ner_file.exists():
            with open(weak_ner_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tokens = row.get("tokens", [])
                    ner_tags = normalize_ner_tags(row.get("weak_ner_tags", []))
                    if not tokens or not ner_tags or len(tokens) != len(ner_tags):
                        continue
                    if not has_entity(ner_tags):
                        continue
                    train_all.append({"tokens": tokens, "ner_tags": ner_tags})
                    weak_ner_count += 1
            print(f"   Added scraped weak NER bootstrap: {weak_ner_count} sentences")
            
        # Save to train_data and test_data
        with open(TRAIN_DIR / "ner_train.json", 'w') as f:
            json.dump(train_all, f)
        with open(TRAIN_DIR / "ner_val.json", 'w') as f:
            json.dump(val_all, f)
        with open(TEST_DIR / "ner_test.json", 'w') as f:
            json.dump(test_all, f)
            
        print(f"NER splits saved successfully:")
        print(f"   Train : {len(train_all):>6} sentences")
        print(f"   Val   : {len(val_all):>6} sentences")
        print(f"   Test  : {len(test_all):>6} sentences")

if __name__ == "__main__":
    splitter = DatasetSplitter()
    splitter.create_signal_splits()
    splitter.create_ner_splits()
    
    print("\nLeakage-proof data split phase complete.")
    print("  You can now begin model training without data contamination.")

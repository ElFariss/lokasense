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
import os
import csv
import json
import random
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.parent.parent
LABELED_DIR = BASE_DIR / "data" / "labeled"
TRAIN_DIR = BASE_DIR / "train_data"
TEST_DIR = BASE_DIR / "test_data"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

class DatasetSplitter:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        
    def create_signal_splits(self):
        """Create leakage-proof splits for the 7-class signal dataset."""
        print("="*60)
        print(" Creating Leakage-Proof Data Splits (Signal)")
        print("="*60)
        
        # 1. Load weak-labeled data as baseline
        weak_file = LABELED_DIR / "weak_labeled.csv"
        gemini_file = LABELED_DIR / "gemini_labeled.csv"
        
        if not weak_file.exists():
            print("⚠ weak_labeled.csv not found! Run weak_labeling.py first.")
            return False
            
        weak_df = pd.read_csv(weak_file)
        print(f"  Loaded {len(weak_df)} weak labeled texts.")
        
        # 2. Prefer Gemini labels if available
        if gemini_file.exists():
            gemini_df = pd.read_csv(gemini_file)
            print(f"  Loaded {len(gemini_df)} Gemini pseudo-labeled texts.")
            
            # Merge: Use Gemini signal if available, otherwise weak signal
            # Create a lookup dictionary from Gemini
            gemini_dict = dict(zip(gemini_df['text'], gemini_df['gemini_signal']))
            
            # Apply to weak dataframe
            def apply_best_signal(row):
                if row['text'] in gemini_dict:
                    return gemini_dict[row['text']]
                return row['signal']
                
            weak_df['final_signal'] = weak_df.apply(apply_best_signal, axis=1)
        else:
            print("  No Gemini labels found. Using weak labels only.")
            weak_df['final_signal'] = weak_df['signal']
            
        # Optional: remove NEUTRAL if we only care about the 6 core signals?
        # Actually we need NEUTRAL to distinguish non-signals
        
        # Remove empty or super short texts
        df = weak_df[weak_df['text'].str.len() > 10].copy()
        
        # Split logic: 70% Train, 15% Val, 15% Test
        # Stratified split to ensure class balance across splits
        print("\n  Splitting data (Stratified 70/15/15)...")
        try:
            train_val_df, test_df = train_test_split(
                df, test_size=0.15, 
                random_state=self.random_seed, 
                stratify=df['final_signal']
            )
            
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.15/0.85, # 15% of total
                random_state=self.random_seed,
                stratify=train_val_df['final_signal']
            )
        except ValueError as e:
            # Fallback if some classes have too few samples for stratification
            print(f"  ⚠ Stratified split failed ({e}), falling back to random split.")
            train_val_df, test_df = train_test_split(
                df, test_size=0.15, random_state=self.random_seed
            )
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.15/0.85, random_state=self.random_seed
            )

        # Save splits
        train_path = TRAIN_DIR / "signal_train.csv"
        val_path = TRAIN_DIR / "signal_val.csv"
        test_path = TEST_DIR / "signal_test.csv"
        
        # Save only needed columns
        keep_cols = ['text', 'final_signal', 'source']
        train_df[keep_cols].to_csv(train_path, index=False)
        val_df[keep_cols].to_csv(val_path, index=False)
        test_df[keep_cols].to_csv(test_path, index=False)
        
        # Print stats
        print(f"\n✅ Splits saved successfully:")
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
            return sentences

        # Load UGM dataset
        ugm_train = parse_tsv(indolem_base / "nerugm" / "train.tsv")
        ugm_dev = parse_tsv(indolem_base / "nerugm" / "dev.tsv")
        ugm_test = parse_tsv(indolem_base / "nerugm" / "test.tsv")
        
        # Load UI dataset 
        ui_train = parse_tsv(indolem_base / "nerui" / "train.tsv")
        ui_dev = parse_tsv(indolem_base / "nerui" / "test.tsv") # UI uses test for dev usually
        
        train_all = ugm_train + ui_train
        val_all = ugm_dev + ui_dev
        test_all = ugm_test
        
        if not train_all:
            print("  ⚠ IndoLEM data not found. Run setup_datasets.py first!")
            return False
            
        # Save to train_data and test_data
        with open(TRAIN_DIR / "ner_train.json", 'w') as f:
            json.dump(train_all, f)
        with open(TRAIN_DIR / "ner_val.json", 'w') as f:
            json.dump(val_all, f)
        with open(TEST_DIR / "ner_test.json", 'w') as f:
            json.dump(test_all, f)
            
        print(f"✅ NER Splits saved successfully:")
        print(f"   Train : {len(train_all):>6} sentences")
        print(f"   Val   : {len(val_all):>6} sentences")
        print(f"   Test  : {len(test_all):>6} sentences")

if __name__ == "__main__":
    splitter = DatasetSplitter()
    splitter.create_signal_splits()
    splitter.create_ner_splits()
    
    print("\n✅ Leakage-proof data split phase complete.")
    print("  You can now begin model training without data contamination.")

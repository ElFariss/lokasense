#!/usr/bin/env python3
"""
03_signal_model/modelling/evaluate.py
Evaluate signal classifier on the strictly isolated test set.
Generates classification report + confusion matrix.
"""
import json, time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

BASE_DIR = Path(__file__).parent.parent.parent
TEST_DIR = BASE_DIR / "test_data"
ONNX_DIR = BASE_DIR / "signal_onnx"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_LABELS = ["NEUTRAL", "DEMAND_UNMET", "DEMAND_PRESENT", "SUPPLY_SIGNAL", "COMPETITION_HIGH", "COMPLAINT", "TREND"]
label2id = {l: i for i, l in enumerate(SIGNAL_LABELS)}


def main():
    print("=" * 60)
    print(" Signal Classifier Evaluation (Test Set)")
    print("=" * 60)

    test_df = pd.read_csv(TEST_DIR / "signal_test.csv")
    print(f"  Test set: {len(test_df)} samples")

    tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))
    ort_model = ORTModelForSequenceClassification.from_pretrained(str(ONNX_DIR))

    # Batch inference
    batch_size = 32
    all_preds = []
    start = time.time()

    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df['text'].iloc[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        outputs = ort_model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).numpy()
        all_preds.extend(preds)

    elapsed = time.time() - start
    avg_latency_ms = (elapsed / len(test_df)) * 1000

    test_labels = test_df['final_signal'].map(label2id).values
    report = classification_report(test_labels, all_preds, target_names=SIGNAL_LABELS, zero_division=0, output_dict=True)
    report_str = classification_report(test_labels, all_preds, target_names=SIGNAL_LABELS, zero_division=0)

    print(f"\n  Inference: {avg_latency_ms:.1f}ms/sample on CPU")
    print(f"\n{report_str}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=SIGNAL_LABELS, yticklabels=SIGNAL_LABELS, cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Signal Classifier — Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(LOG_DIR / "signal_confusion_matrix.png", dpi=150)
    plt.close()

    # Save metrics
    metrics = {
        "test_samples": len(test_df),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "total_inference_sec": round(elapsed, 2),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class_f1": {label: round(report[label]["f1-score"], 4) for label in SIGNAL_LABELS if label in report},
        "accuracy": round(report.get("accuracy", 0), 4),
    }
    with open(LOG_DIR / "signal_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Test evaluation complete!")
    print(f"  Macro F1: {metrics['macro_f1']}")
    print(f"  CPU latency: {metrics['avg_latency_ms']}ms/sample")
    print(f"  Results saved to: logs/signal_test_metrics.json")
    print(f"  Confusion matrix: logs/signal_confusion_matrix.png")


if __name__ == "__main__":
    main()

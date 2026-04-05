#!/usr/bin/env python3
"""
02_ner_model/modelling/evaluate.py
Evaluate ONNX NER model on test set.
"""
import json, time
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from seqeval.metrics import classification_report

BASE_DIR = Path(__file__).parent.parent.parent
TEST_DIR = BASE_DIR / "test_data"
ONNX_DIR = BASE_DIR / "ner_onnx"
LOG_DIR = BASE_DIR / "logs"

def main():
    print("=" * 60)
    print(" NER Model Evaluation (Test Set)")
    print("=" * 60)

    test_file = TEST_DIR / "ner_test.json"
    if not test_file.exists():
        print("Test data not found.")
        return

    with open(test_file) as f:
        test_data = json.load(f)

    # Load label mapping from training output
    with open(BASE_DIR / "models" / "ner_base" / "label_mapping.json") as f:
        mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))
    ort_model = ORTModelForTokenClassification.from_pretrained(str(ONNX_DIR))

    print(f"  Testing on {len(test_data)} samples")
    
    true_labels = []
    pred_labels = []
    
    start = time.time()
    
    for item in test_data:
        tokens = item["tokens"]
        labels = item["ner_tags"]
        
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
        word_ids = inputs.word_ids()
        
        outputs = ort_model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)[0].numpy()
        
        # Align predictions back to words
        sent_true = []
        sent_pred = []
        
        prev_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                # new word
                sent_true.append(labels[word_id])
                pred_label = id2label[preds[i]]
                sent_pred.append(pred_label)
            prev_word_id = word_id
            
        true_labels.append(sent_true)
        pred_labels.append(sent_pred)

    elapsed = time.time() - start
    avg_ms = (elapsed / len(test_data)) * 1000

    report_dict = classification_report(true_labels, pred_labels, output_dict=True)
    report_str = classification_report(true_labels, pred_labels)

    print(f"\n  Inference: {avg_ms:.1f}ms/sample on CPU")
    print(f"\n{report_str}")

    metrics = {
        "test_samples": len(test_data),
        "avg_latency_ms": round(avg_ms, 1),
        "total_inference_sec": round(elapsed, 2),
        "f1": round(report_dict["micro avg"]["f1-score"], 4),
    }
    with open(LOG_DIR / "ner_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to logs/ner_test_metrics.json")


if __name__ == "__main__":
    main()

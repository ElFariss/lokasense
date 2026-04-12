#!/usr/bin/env python3
"""
05_explainability/modelling/explain.py
Per-sample LIME explanation. Satisfies Track C-1, C-2, C-3.
"""
import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from lime.lime_text import LimeTextExplainer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from processing.lime_wrapper import LIMEPredictor, SIGNAL_LABELS

BASE_DIR = Path(__file__).parent.parent.parent
ONNX_DIR = BASE_DIR / "signal_onnx"
TEST_DIR = BASE_DIR / "test_data"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def explain_single(predictor, text, num_features=3, num_samples=500):
    """Generate LIME explanation for a single text (Track C-2 compliant)."""
    probs = predictor.predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    predicted_signal = SIGNAL_LABELS[pred_idx]
    confidence = float(probs[pred_idx])

    explainer = LimeTextExplainer(class_names=SIGNAL_LABELS)
    exp = explainer.explain_instance(
        text, predictor.predict_proba,
        labels=[pred_idx], num_features=num_features, num_samples=num_samples
    )

    top_features = []
    for token, weight in exp.as_list(label=pred_idx):
        direction = "↑" if weight > 0 else "↓"
        impact = "memperkuat" if weight > 0 else "memperlemah"
        top_features.append({
            "token": token,
            "weight": round(weight, 4),
            "direction": direction,
            "explanation": f"Kata '{token}' {impact} prediksi {predicted_signal}."
        })

    return {
        "text": text,
        "predicted_signal": predicted_signal,
        "confidence": round(confidence, 4),
        "top_3_features": top_features,
        "all_probabilities": {SIGNAL_LABELS[i]: round(float(probs[i]), 4) for i in range(len(SIGNAL_LABELS))},
    }


def main():
    print("=" * 60)
    print(" LIME Explainability — Track C Compliance")
    print("=" * 60)

    predictor = LIMEPredictor(ONNX_DIR)

    # Explain a subset of the test set
    test_df = pd.read_csv(TEST_DIR / "signal_test.csv")
    sample_size = min(20, len(test_df))
    sample_df = test_df.sample(sample_size, random_state=42)

    explanations = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        print(f"  [{i+1}/{sample_size}] Explaining: {row['text'][:60]}...")
        start = time.time()
        result = explain_single(predictor, row['text'])
        result['true_label'] = row.get('final_signal', '')
        result['inference_ms'] = round((time.time() - start) * 1000, 1)
        explanations.append(result)

    # Save
    with open(LOG_DIR / "lime_explanations.json", "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    print(f"\n{len(explanations)} LIME explanations saved to logs/lime_explanations.json")

    # Print a sample
    if explanations:
        e = explanations[0]
        print(f"\n  Sample Explanation:")
        print(f"  Text: {e['text'][:80]}...")
        print(f"  Predicted: {e['predicted_signal']} ({e['confidence']:.2f})")
        print(f"  Top 3 Features:")
        for feat in e['top_3_features']:
            print(f"    {feat['direction']} {feat['token']}: {feat['weight']:.4f}")
            print(f"      {feat['explanation']}")


if __name__ == "__main__":
    main()

# %% [markdown]
# # LokaSense (Pasarint) — Inference & Explainability Pipeline
# 
# **Hackathon: FindIT! 2026 — Track C (The Explainable Oracle)**  
# This notebook represents the end-to-end local inference pipeline.
# 
# ### Constraint Compliance Checklist:
# - [x] **C-1: Explainability (SHAP/LIME):** Integrated `lime-text` for the signal classifier.
# - [x] **C-2: Top-3 Features + Direction:** Output format strictly includes top-3 LIME tokens and their up/down weights.
# - [x] **C-5 & Gen: Offline Total & CPU-Only:** Uses INT8 ONNX models with zero external API calls.
# - [x] **Bonus (Track B): PII Filter:** Redacts NIK, phone, email, and bank account before inference.

# %% [code]
import os
import re
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# PyTorch (for tokenizer only) and ONNX
import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# Explainability
from lime.lime_text import LimeTextExplainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = Path()
ONNX_DIR = BASE_DIR / "signal_onnx"

SIGNAL_LABELS = [
    "NEUTRAL", "DEMAND_UNMET", "DEMAND_PRESENT", 
    "SUPPLY_SIGNAL", "COMPETITION_HIGH", "COMPLAINT", "TREND"
]

# %% [markdown]
# ## 1. Load ONNX Models (Offline / Local)
# Loading the INT8 quantized model using Optimum's ORT wrapper.

# %% [code]
print("Loading Tokenizer and ONNX Model...")

# In a real environment, this would point to the exported ONNX dir
# We use a try/except so the structure works even if training hasn't run yet
try:
    tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
    ort_model = ORTModelForSequenceClassification.from_pretrained(ONNX_DIR)
    print("✅ Model loaded successfully from local storage.")
except Exception as e:
    print(f"⚠ Warning: Could not load ONNX model (Run training.ipynb first). Error: {e}")
    # Mock fallback for demonstration structure
    tokenizer = None
    ort_model = None

# %% [markdown]
# ## 2. Core Inference and Explainability Engine (LIME)
# We wrap the ONNX model in a function that returns probability arrays, which LIME requires.

# %% [code]
def predict_proba(texts):
    """Wrapper for LIME to get prediction probabilities from ONNX."""
    if ort_model is None:
        # Mock probabilities if model missing
        probs = np.random.dirichlet(np.ones(len(SIGNAL_LABELS)), size=len(texts))
        return probs
        
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    outputs = ort_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.detach().numpy()

def analyze_market_signal(text, num_features=3):
    """
    Run inference & LIME explainability on a single text.
    Satisfies Track C-1 and C-2 (Top-3 features + direction).
    """
    start_time = time.time()
    
    # 1. Base Prediction
    probs = predict_proba([text])[0]
    pred_idx = np.argmax(probs)
    predicted_signal = SIGNAL_LABELS[pred_idx]
    confidence = float(probs[pred_idx])
    
    # 2. LIME Explainability (Track C)
    explainer = LimeTextExplainer(class_names=SIGNAL_LABELS)
    
    # We explain the prediction for the predicted class
    exp = explainer.explain_instance(
        text, 
        predict_proba, 
        labels=[pred_idx], 
        num_features=num_features,
        num_samples=100  # Trade-off: speed vs explainability resolution
    )
    
    # Extract top features
    # exp.as_list returns tuples of (token, weight)
    explanation_list = exp.as_list(label=pred_idx)
    
    # Format output
    top_features = []
    readable_sentences = []
    
    for token, weight in explanation_list:
        direction = "↑" if weight > 0 else "↓"
        impact = "strengthens" if weight > 0 else "weakens"
        top_features.append(f"{token} ({direction}{abs(weight):.3f})")
        
        readable_sentences.append(f"Kata '{token}' {impact} prediksi bahwa ini adalah {predicted_signal}.")

    latency = time.time() - start_time
    
    return {
        "text": text,
        "predicted_signal": predicted_signal,
        "confidence": confidence,
        "top_3_features": top_features,
        "explanation_summary": " | ".join(readable_sentences),
        "inference_time_sec": round(latency, 3),
        "lime_html": exp.as_html()  # Can be rendered in notebook
    }

# %% [markdown]
# ## 3. PII Filtering (Bonus / Track B Compliance)

# %% [code]
def redact_pii(text):
    """Redact sensitive information (PII) before it hits the model."""
    # NIK (16 digits)
    text = re.sub(r'\b\d{16}\b', '[NIK_REDACTED]', text)
    # Phone numbers
    text = re.sub(r'(\+62|08)\d{8,12}', '[PHONE_REDACTED]', text)
    # Emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', text)
    # Bank Account (heuristics: 10-16 digits often prefixed with 'rek' or bank names)
    text = re.sub(r'(?i)(rek|rekening|bca|mandiri|bni|bri)\s*\d{10,16}\b', r'\1 [BANK_REDACTED]', text)
    return text

# %% [markdown]
# ## 4. End-to-End Pipeline Demonstration
# Testing the strict format requirements of Track C-2.

# %% [code]
test_texts = [
    "Di Lowokwaru +6281234567890 belum ada dimsum yang seenak ini. Bikin kangen",
    "Mahal banget, porsinya kedikitan, pelayanan lambat",
    "Udah ada 3 outlet mie gacoan yang buka di jalan ini"
]

print("="*70)
print(" 🚀 LokaSense Local Inference & Explainability Engine (Track C)")
print("="*70)

for original_text in test_texts:
    # 1. PII Redaction
    safe_text = redact_pii(original_text)
    
    # 2. Inference + LIME
    result = analyze_market_signal(safe_text)
    
    print("\n📝 PII-Filtered Text:", result["text"])
    print(f"📊 Prediction: {result['predicted_signal']} (Confidence: {result['confidence']:.2f})")
    print(f"⚡ Latency: {result['inference_time_sec']}s (CPU Only)")
    print("🔍 Explainability (Top 3 Features):")
    for feat in result["top_3_features"]:
        print(f"   - {feat}")
    print(f"💬 Human Readable: {result['explanation_summary']}")

# Note: In a UI environment, we would also render result["lime_html"] using IPython.display.HTML

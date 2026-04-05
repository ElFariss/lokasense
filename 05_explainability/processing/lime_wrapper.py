#!/usr/bin/env python3
"""
05_explainability/processing/lime_wrapper.py
LIME predict_proba wrapper for ONNX inference. Track C-1 compliance.
"""
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

SIGNAL_LABELS = ["NEUTRAL", "DEMAND_UNMET", "DEMAND_PRESENT", "SUPPLY_SIGNAL", "COMPETITION_HIGH", "COMPLAINT", "TREND"]


class LIMEPredictor:
    """LIME-compatible wrapper around the ONNX signal classifier."""

    def __init__(self, onnx_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
        self.model = ORTModelForSequenceClassification.from_pretrained(str(onnx_dir))

    def predict_proba(self, texts):
        """Return probability array for LIME. Shape: (n_texts, n_classes)."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.detach().numpy()

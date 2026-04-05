# 05 — Explainability (Track C Compliance)

## Track C Constraints Satisfied
| ID | Constraint | Implementation |
|----|-----------|----------------|
| C-1 | Explainability (SHAP/LIME) | LimeTextExplainer on signal classifier |
| C-2 | Top 3 features + direction | LIME token weights with ↑/↓ indicators |
| C-3 | Anti-Black Box | Attention weights + feature importance plots |

## Directory Structure
```
05_explainability/
├── README.md             ← This file
├── processing/
│   └── lime_wrapper.py   ← LIME predict_proba wrapper for ONNX
└── modelling/
    ├── explain.py         ← Per-sample LIME explanation
    └── global_importance.py ← Global feature importance across test set
```

## Output Format (Track C-2 Compliance)
Every prediction includes:
1. Predicted signal class
2. Confidence score (softmax probability)
3. **Top 3 influential tokens with direction (+/−)**
4. Human-readable explanation sentence
5. LIME HTML visualization (for notebook rendering)

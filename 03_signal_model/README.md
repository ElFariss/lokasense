# 03 — Signal Classifier (7-Class Market Signal)

## Architecture
- **Base:** `indobenchmark/indobert-base-p1` (124.5M params)
- **Task:** Sequence Classification — 7 market signal classes
- **Export:** ONNX INT8 dynamic quantization for CPU inference

## Directory Structure
```
03_signal_model/
├── README.md           ← This file
├── dataset/
│   ├── weak_label.py   ← Keyword-based weak labeling for 7 signals
│   ├── gemini_label.py ← Gemini 2.5 Flash pseudo-labeling
│   └── split.py        ← Leakage-proof train/val/test split
├── processing/
│   ├── normalize.py    ← Text cleaning + slang normalization
│   └── eda.py          ← Exploratory Data Analysis with stats
└── modelling/
    ├── train.py         ← Fine-tuning with weighted CrossEntropy
    ├── export_onnx.py   ← ONNX export + INT8 quantization
    └── evaluate.py      ← Test set evaluation + confusion matrix
```

## Signal Classes
| Signal | Description | Example |
|--------|-------------|---------|
| DEMAND_UNMET | Unmet demand (strongest opportunity) | "Di sini belum ada dimsum" |
| DEMAND_PRESENT | Validates existing demand | "Bakso di sini enak banget" |
| SUPPLY_SIGNAL | Factual supply observation | "Udah ada 3 outlet Mixue" |
| COMPETITION_HIGH | Perceived oversaturation | "Banyak banget cafe dimana-mana" |
| COMPLAINT | Quality/service complaint | "Mahal, porsi kecil, jorok" |
| TREND | Viral/trending signal | "Lagi viral di TikTok!" |
| NEUTRAL | No actionable signal | "Buka jam 8 pagi" |

## Training Hyperparameters
- lr: 2e-5
- batch_size: 16
- epochs: 5
- max_length: 128
- loss: weighted CrossEntropyLoss (class imbalance handling)
- eval: macro F1
- early stopping on val macro-F1

## Data Pipeline
```
Raw text → Text normalization → Weak labeling → Gemini pseudo-label → Split → Train
```

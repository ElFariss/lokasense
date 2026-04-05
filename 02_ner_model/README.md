# 02 — NER Model (Named Entity Recognition)

## Architecture
- **Base:** `indobenchmark/indobert-base-p1` (124.5M params)
- **Task:** Token Classification — extract LOC, ORG, PER entities from Indonesian text
- **Export:** ONNX INT8 dynamic quantization for CPU inference

## Directory Structure
```
02_ner_model/
├── README.md          ← This file
├── dataset/
│   └── prepare.py     ← Load IndoLEM NER data, format for HF Trainer
├── processing/
│   └── preprocess.py  ← Token alignment, label mapping
└── modelling/
    ├── train.py        ← Fine-tuning with HF Trainer
    ├── export_onnx.py  ← ONNX export + INT8 quantization
    └── evaluate.py     ← Test set evaluation with seqeval
```

## Data Sources
- IndoLEM NER-UGM (train/dev/test TSV)
- IndoLEM NER-UI (train/test TSV)
- Scraped weak NER bootstrap from `data/scraped/ner_bootstrap.jsonl`

## Labels
| Tag | Entity Type |
|-----|-------------|
| B-PER / I-PER | Person name |
| B-LOC / I-LOC | Location (kecamatan, kota, etc.) |
| B-ORG / I-ORG | Organization |
| O | Outside / no entity |

## Training Hyperparameters
- lr: 2e-5
- batch_size: 16
- epochs: 5
- max_length: 128
- eval: seqeval F1 per entity
- early stopping on val F1

#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import nbformat as nbf

BASE_DIR = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "Palantir Cabang Malang_Tahap2_FindIT2026"
PACKAGE_DIR = BASE_DIR / PACKAGE_NAME

NOTEBOOKS_DIR = PACKAGE_DIR / "notebooks"
MODELS_DIR = PACKAGE_DIR / "models"
DATASETS_DIR = PACKAGE_DIR / "datasets"
REPORTS_DIR = PACKAGE_DIR / "reports"
CODE_DIR = PACKAGE_DIR / "code"
OUTPUTS_DIR = PACKAGE_DIR / "example_outputs"

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints", ".DS_Store")


def reset_package_dir() -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    for path in [NOTEBOOKS_DIR, MODELS_DIR, DATASETS_DIR, REPORTS_DIR, CODE_DIR, OUTPUTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_tree(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest, ignore=IGNORE, dirs_exist_ok=True)


def create_signal_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# training_signal1.ipynb\n\n"
            "Notebook ini merepresentasikan **model utama** LokaSense: klasifikasi sinyal pasar "
            "berbahasa Indonesia untuk pemetaan peluang bisnis UMKM."
        ),
        nbf.v4.new_markdown_cell(
            "## Peran Model\n\n"
            "- Nama kemasan: `model_signal1`\n"
            "- Fungsi: mengklasifikasikan teks menjadi 7 kelas sinyal pasar\n"
            "- Backbone: IndoBERT\n"
            "- Pipeline training penuh tetap diringkas di `training_main.ipynb` karena repo dijaga "
            "dengan alur notebook-terpadu."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n\n"
            "BASE_DIR = Path.cwd().resolve().parent\n"
            "metrics_path = BASE_DIR / 'reports' / 'signal_test_metrics_pytorch.json'\n"
            "train_metrics_path = BASE_DIR / 'reports' / 'signal_training_metrics.json'\n"
            "dataset_path = BASE_DIR / 'datasets' / 'signal' / 'weak_labeled.csv'\n\n"
            "print('Signal model folder :', BASE_DIR / 'models' / 'model_signal1')\n"
            "print('Dataset file        :', dataset_path)\n"
            "print('Metrics file        :', metrics_path)\n\n"
            "if train_metrics_path.exists():\n"
            "    print('\\nTraining metrics:')\n"
            "    print(json.dumps(json.loads(train_metrics_path.read_text()), indent=2, ensure_ascii=False))\n\n"
            "if metrics_path.exists():\n"
            "    print('\\nTest metrics:')\n"
            "    print(json.dumps(json.loads(metrics_path.read_text()), indent=2, ensure_ascii=False))\n\n"
            "if dataset_path.exists():\n"
            "    df = pd.read_csv(dataset_path)\n"
            "    label_col = 'final_signal' if 'final_signal' in df.columns else 'signal'\n"
            "    print('\\nWeak-labeled distribution:')\n"
            "    print(df[label_col].value_counts(dropna=False).to_string())\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Catatan Eksekusi\n\n"
            "Notebook ini disertakan untuk memenuhi ketentuan penamaan model utama. "
            "Ringkasan pipeline end-to-end, EDA, preprocessing, training, evaluasi, dan readiness gate "
            "ada di `training_main.ipynb`."
        ),
    ]
    nbf.write(nb, NOTEBOOKS_DIR / "training_signal1.ipynb")


def create_ner_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# training_ner2.ipynb\n\n"
            "Notebook ini merepresentasikan **model pendukung** LokaSense: Named Entity Recognition "
            "untuk ekstraksi lokasi pada teks Indonesia."
        ),
        nbf.v4.new_markdown_cell(
            "## Peran Model\n\n"
            "- Nama kemasan: `model_ner2`\n"
            "- Fungsi: mendeteksi entitas lokasi dan entitas terkait pada teks\n"
            "- Backbone: IndoBERT\n"
            "- Model ini mendukung resolusi lokasi pada runtime `analyze.py`."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n\n"
            "BASE_DIR = Path.cwd().resolve().parent\n"
            "metrics_path = BASE_DIR / 'reports' / 'ner_test_metrics_pytorch.json'\n"
            "train_metrics_path = BASE_DIR / 'reports' / 'ner_training_metrics.json'\n"
            "dataset_path = BASE_DIR / 'datasets' / 'ner' / 'ner_test.json'\n\n"
            "print('NER model folder    :', BASE_DIR / 'models' / 'model_ner2')\n"
            "print('Dataset file        :', dataset_path)\n"
            "print('Metrics file        :', metrics_path)\n\n"
            "if train_metrics_path.exists():\n"
            "    print('\\nTraining metrics:')\n"
            "    print(json.dumps(json.loads(train_metrics_path.read_text()), indent=2, ensure_ascii=False))\n\n"
            "if metrics_path.exists():\n"
            "    print('\\nTest metrics:')\n"
            "    print(json.dumps(json.loads(metrics_path.read_text()), indent=2, ensure_ascii=False))\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Catatan Eksekusi\n\n"
            "Notebook ringkasan ini mendampingi `training_main.ipynb`. "
            "Pipeline training yang dipelihara tetap notebook utama agar artefak training dan evaluasi "
            "tetap konsisten."
        ),
    ]
    nbf.write(nb, NOTEBOOKS_DIR / "training_ner2.ipynb")


def create_package_readme() -> None:
    readme = f"""# {PACKAGE_NAME}

Folder ini disiapkan sebagai bundel teknis tahap 2 dan berisi notebook, model, dataset, kode runtime, serta laporan evaluasi yang relevan untuk LokaSense.

## Struktur Isi

- `notebooks/training_main.ipynb`
  Ringkasan pipeline utama yang dipelihara. Ini adalah notebook end-to-end untuk EDA, preprocessing, training, evaluasi, scoring, dan readiness check.
- `notebooks/training_signal1.ipynb`
  Notebook model utama untuk klasifikasi sinyal pasar.
- `notebooks/training_ner2.ipynb`
  Notebook model pendukung untuk NER lokasi.
- `notebooks/inference.ipynb`
  Notebook inference lokal.
- `models/model_signal1/`
  Checkpoint Hugging Face model utama.
- `models/model_ner2/`
  Checkpoint Hugging Face model pendukung.
- `datasets/`
  Dataset inti untuk signal, NER, runtime airgap, data sosial, POI, dan file pendukung.
- `code/`
  Kode runtime yang diperlukan untuk analisis query sampai heatmap.
- `reports/`
  Metrik training, metrik evaluasi, readiness report, dan file scoring.
- `example_outputs/`
  Contoh output heatmap hasil runtime airgap.

## Catatan Penamaan Model

Format asli model proyek ini berbasis checkpoint Hugging Face, sehingga model dikemas sebagai folder, bukan satu file tunggal. Untuk tetap mengikuti ketentuan:

- model utama: `models/model_signal1/`
- model pendukung: `models/model_ner2/`

Masing-masing folder sudah memuat `config.json`, tokenizer, dan `model.safetensors`.

## Cara Buka Cepat

1. Buka `notebooks/training_main.ipynb` untuk melihat pipeline utama.
2. Buka `notebooks/inference.ipynb` untuk demo inference.
3. Jalankan `code/analyze.py` jika ingin memakai runtime query-to-heatmap.
4. Lihat `reports/airgap_production_readiness.json` untuk status kesiapan offline terbaru.

## Dependensi

Dependensi proyek disertakan dalam `requirements.txt`.

## Status Validasi Penting

- Secara operasional, runtime airgap sudah lolos acceptance checks.
- Klaim production-ready penuh masih menunggu **review manual manusia** pada `datasets/signal/signal_test_manual.csv`.
- File readiness offline membedakan:
  - `operational_status`
  - `status`

Itu sengaja dipisahkan agar tidak overclaim.
"""
    (PACKAGE_DIR / "README.md").write_text(readme, encoding="utf-8")


def create_datasets_readme() -> None:
    readme = """# Dataset Guide

Folder ini merangkum dataset yang dipakai atau dibutuhkan untuk menjalankan pipeline LokaSense dalam bentuk yang relevan untuk submission.

## Struktur

- `signal/`
  - `weak_labeled.csv`: korpus signal hasil weak labeling utama.
  - `model_pseudo_augmented.csv`: tambahan pseudolabel lokal berbasis IndoBERT.
  - `gemini_augmented.csv`: tambahan pseudolabel Gemini opsional.
  - `signal_bootstrap.csv`: bootstrap korpus signal dari scraping.
  - `signal_test.csv`: test split bawaan.
  - `signal_test_manual.csv`: gold review set untuk validasi yang lebih ketat.

- `ner/`
  - `ner_bootstrap.jsonl`: bootstrap data NER hasil heuristik/gazetteer.
  - `ner_test.json`: set evaluasi NER.

- `runtime/`
  - `airgap_corpus.csv`: korpus offline gabungan untuk mode airgap.
  - `airgap_manifest.json`: manifest sumber korpus airgap.

- `social/`
  - `tiktok_data.csv`: hasil scraping TikTok yang dipakai dalam pipeline.

- `poi/`
  - `overpass_poi.csv`: data POI OpenStreetMap untuk scoring dan penalti franchise.

- `slang/`
  - `slang_dict.json`: kamus normalisasi slang Indonesia.

## Catatan Validasi

- `signal_test_manual.csv` saat ini berisi label draft AI-assisted untuk memudahkan packaging.
- Untuk klaim evaluasi final yang benar-benar manual, file itu perlu direview manusia.

## Catatan Biaya

Bundel ini disiapkan untuk path offline dan tidak membutuhkan API berbayar saat dipakai dengan mode airgap.
"""
    (DATASETS_DIR / "README.md").write_text(readme, encoding="utf-8")


def create_manifest() -> None:
    payload = {
        "package_name": PACKAGE_NAME,
        "notebooks": [
            "notebooks/training_main.ipynb",
            "notebooks/training_signal1.ipynb",
            "notebooks/training_ner2.ipynb",
            "notebooks/inference.ipynb",
        ],
        "models": [
            "models/model_signal1",
            "models/model_ner2",
        ],
        "reports": [
            "reports/signal_training_metrics.json",
            "reports/signal_test_metrics_pytorch.json",
            "reports/ner_training_metrics.json",
            "reports/ner_test_metrics_pytorch.json",
            "reports/production_readiness.json",
            "reports/airgap_production_readiness.json",
            "reports/opportunity_scores.csv",
            "reports/opportunity_scores.json",
        ],
    }
    (PACKAGE_DIR / "MANIFEST.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_package() -> None:
    reset_package_dir()

    copy_file(BASE_DIR / "training.ipynb", NOTEBOOKS_DIR / "training_main.ipynb")
    copy_file(BASE_DIR / "inference.ipynb", NOTEBOOKS_DIR / "inference.ipynb")
    create_signal_notebook()
    create_ner_notebook()

    copy_tree(BASE_DIR / "models" / "signal_base", MODELS_DIR / "model_signal1")
    copy_tree(BASE_DIR / "models" / "ner_base", MODELS_DIR / "model_ner2")

    dataset_map = {
        DATASETS_DIR / "signal" / "weak_labeled.csv": BASE_DIR / "data" / "labeled" / "weak_labeled.csv",
        DATASETS_DIR / "signal" / "model_pseudo_augmented.csv": BASE_DIR / "data" / "labeled" / "model_pseudo_augmented.csv",
        DATASETS_DIR / "signal" / "gemini_augmented.csv": BASE_DIR / "data" / "labeled" / "gemini_augmented.csv",
        DATASETS_DIR / "signal" / "signal_bootstrap.csv": BASE_DIR / "data" / "scraped" / "signal_bootstrap.csv",
        DATASETS_DIR / "signal" / "signal_test.csv": BASE_DIR / "test_data" / "signal_test.csv",
        DATASETS_DIR / "signal" / "signal_test_manual.csv": BASE_DIR / "test_data" / "signal_test_manual.csv",
        DATASETS_DIR / "ner" / "ner_bootstrap.jsonl": BASE_DIR / "data" / "scraped" / "ner_bootstrap.jsonl",
        DATASETS_DIR / "ner" / "ner_test.json": BASE_DIR / "test_data" / "ner_test.json",
        DATASETS_DIR / "runtime" / "airgap_corpus.csv": BASE_DIR / "data" / "airgap" / "airgap_corpus.csv",
        DATASETS_DIR / "runtime" / "airgap_manifest.json": BASE_DIR / "data" / "airgap" / "airgap_manifest.json",
        DATASETS_DIR / "social" / "tiktok_data.csv": BASE_DIR / "data" / "social_media" / "tiktok_data.csv",
        DATASETS_DIR / "poi" / "overpass_poi.csv": BASE_DIR / "data" / "poi" / "overpass_poi.csv",
        DATASETS_DIR / "slang" / "slang_dict.json": BASE_DIR / "data" / "slang" / "slang_dict.json",
    }
    for dest, src in dataset_map.items():
        if src.exists():
            copy_file(src, dest)

    reports = [
        "signal_training_metrics.json",
        "signal_test_metrics_pytorch.json",
        "ner_training_metrics.json",
        "ner_test_metrics_pytorch.json",
        "production_readiness.json",
        "airgap_production_readiness.json",
        "opportunity_scores.csv",
        "opportunity_scores.json",
    ]
    for name in reports:
        copy_file(BASE_DIR / "logs" / name, REPORTS_DIR / name)

    outputs = [
        "airgap_malang.html",
        "airgap_surabaya.html",
        "airgap_bandung.html",
        "lokasense_heatmap.html",
    ]
    for name in outputs:
        src = BASE_DIR / "outputs" / name
        if src.exists():
            copy_file(src, OUTPUTS_DIR / name)

    copy_file(BASE_DIR / "requirements.txt", PACKAGE_DIR / "requirements.txt")
    copy_file(BASE_DIR / "DATA.md", PACKAGE_DIR / "DATA.md")

    copy_file(BASE_DIR / "analyze.py", CODE_DIR / "analyze.py")
    copy_tree(BASE_DIR / "06_agent", CODE_DIR / "06_agent")
    copy_tree(BASE_DIR / "common", CODE_DIR / "common")
    copy_tree(BASE_DIR / "04_spatial_engine" / "modelling", CODE_DIR / "04_spatial_engine" / "modelling")
    copy_tree(BASE_DIR / "05_explainability" / "modelling", CODE_DIR / "05_explainability" / "modelling")
    copy_file(BASE_DIR / "scripts" / "check_airgap_readiness.py", CODE_DIR / "scripts" / "check_airgap_readiness.py")
    copy_file(BASE_DIR / "scripts" / "build_manual_test_set.py", CODE_DIR / "scripts" / "build_manual_test_set.py")
    copy_file(BASE_DIR / "scripts" / "build_airgap_corpus.py", CODE_DIR / "scripts" / "build_airgap_corpus.py")
    copy_file(BASE_DIR / "scripts" / "fill_ai_review_gold_set.py", CODE_DIR / "scripts" / "fill_ai_review_gold_set.py")

    create_package_readme()
    create_datasets_readme()
    create_manifest()

    print(f"Created package: {PACKAGE_DIR}")


if __name__ == "__main__":
    build_package()

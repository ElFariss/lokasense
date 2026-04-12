#!/usr/bin/env python3
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_NOTEBOOK = REPO_ROOT / "training.ipynb"
INFERENCE_NOTEBOOK = REPO_ROOT / "inference.ipynb"
KERNEL_NAME = "ugm_hackathon"
DISPLAY_NAME = "Python (UGM_HACKATHON)"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


TRAINING_SETUP = """
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dotenv import load_dotenv
from IPython.display import display

REPO_ROOT = Path.cwd()
assert (REPO_ROOT / "01_data_collection").exists(), "Please run this notebook from the repo root."
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SPATIAL_MODELLING_DIR = REPO_ROOT / "04_spatial_engine" / "modelling"
if str(SPATIAL_MODELLING_DIR) not in sys.path:
    sys.path.insert(0, str(SPATIAL_MODELLING_DIR))

import scoring as spatial_scoring
import heatmap as spatial_heatmap

from common.bootstrap_utils import build_ner_bootstrap_rows, build_signal_bootstrap_rows
from common.location_resolution import LocationResolver
from common.text_normalization import language_scores, strip_emoji

load_dotenv(REPO_ROOT / ".env")

pd.set_option("display.max_colwidth", 160)
pd.set_option("display.max_rows", 200)
sns.set_theme(style="whitegrid")

SCRAPING_PYTHON = REPO_ROOT / ".venv_scraping" / "bin" / "python"
SOCIAL_SOURCE_FILES = {
    "tiktok": REPO_ROOT / "data" / "social_media" / "tiktok_data.csv",
    "instagram": REPO_ROOT / "data" / "social_media" / "instagram_data.csv",
    "x": REPO_ROOT / "data" / "social_media" / "x_data.csv",
}


def count_existing_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
        return int(len(df))
    except Exception:
        return 0


SOCIAL_REFRESH_SPECS = [
    {
        "platform": "tiktok",
        "source_file": SOCIAL_SOURCE_FILES["tiktok"],
        "min_rows": int(os.getenv("NOTEBOOK_MIN_TIKTOK_ROWS", "220")),
        "max_queries": int(os.getenv("NOTEBOOK_TIKTOK_MAX_QUERIES", "180")),
        "max_saved_rows": int(os.getenv("NOTEBOOK_TIKTOK_MAX_ROWS", "240")),
    },
    {
        "platform": "instagram",
        "source_file": SOCIAL_SOURCE_FILES["instagram"],
        "min_rows": int(os.getenv("NOTEBOOK_MIN_INSTAGRAM_ROWS", "80")),
        "max_queries": int(os.getenv("NOTEBOOK_INSTAGRAM_MAX_QUERIES", "120")),
        "max_saved_rows": int(os.getenv("NOTEBOOK_INSTAGRAM_MAX_ROWS", "100")),
    },
    {
        "platform": "x",
        "source_file": SOCIAL_SOURCE_FILES["x"],
        "min_rows": int(os.getenv("NOTEBOOK_MIN_X_ROWS", "80")),
        "max_queries": int(os.getenv("NOTEBOOK_X_MAX_QUERIES", "120")),
        "max_saved_rows": int(os.getenv("NOTEBOOK_X_MAX_ROWS", "100")),
    },
]
for spec in SOCIAL_REFRESH_SPECS:
    spec["existing_rows"] = count_existing_rows(spec["source_file"])

FORCE_SOCIAL_REFRESH = os.getenv("FORCE_NOTEBOOK_SOCIAL_REFRESH", "0") == "1"
SOCIAL_REFRESH_PLAN = [
    spec
    for spec in SOCIAL_REFRESH_SPECS
    if FORCE_SOCIAL_REFRESH or int(spec["existing_rows"]) < int(spec["min_rows"])
]
INCLUDE_GOOGLE_MAPS_CACHE = os.getenv("INCLUDE_GMAPS_CACHE", "0") == "1"
RUN_GEMINI_AUGMENTATION = os.getenv("ENABLE_NOTEBOOK_GEMINI", "0") == "1"
GEMINI_MAX_SAMPLES = int(os.getenv("NOTEBOOK_GEMINI_MAX_SAMPLES", "1200"))
FORCE_GEMINI_REFRESH = os.getenv("FORCE_NOTEBOOK_GEMINI_REFRESH", "0") == "1"
RUN_MODEL_PSEUDOLABEL = os.getenv("ENABLE_NOTEBOOK_MODEL_PSEUDOLABEL", "0") == "1"
MODEL_PSEUDOLABEL_MAX_SAMPLES = int(os.getenv("NOTEBOOK_MODEL_PSEUDOLABEL_MAX_SAMPLES", "600"))
FORCE_MODEL_PSEUDOLABEL_REFRESH = os.getenv("FORCE_NOTEBOOK_MODEL_PSEUDOLABEL_REFRESH", "0") == "1"
GOOGLE_MAPS_REFRESH_REQUESTED = os.getenv("ENABLE_GOOGLE_MAPS_REFRESH", "0") == "1"
if GOOGLE_MAPS_REFRESH_REQUESTED:
    print("Notebook keeps Google Maps refresh disabled. Run collect_gmaps_reviews.py --confirm-billable manually if you truly need it.")

def build_social_refresh_args(spec: dict[str, object]) -> list[str]:
    return [
        str(SCRAPING_PYTHON if SCRAPING_PYTHON.exists() else Path(sys.executable)),
        "01_data_collection/collect_social_bootstrap.py",
        "--platform",
        str(spec["platform"]),
        "--max-queries",
        str(spec["max_queries"]),
        "--max-per-query",
        "8",
        "--max-saved-rows",
        str(spec["max_saved_rows"]),
        "--headless",
        "--query-delay",
        "0.6",
    ]

SIGNAL_LABELS = [
    "NEUTRAL",
    "DEMAND_UNMET",
    "DEMAND_PRESENT",
    "SUPPLY_SIGNAL",
    "COMPETITION_HIGH",
    "COMPLAINT",
    "TREND",
]

READINESS_THRESHOLDS = {
    "signal_train_rows": 300,
    "signal_eval_macro_f1": 0.55,
    "signal_gold_macro_f1": 0.50,
    "ner_eval_micro_f1": 0.75,
    "opportunity_groups": 5,
}
MANUAL_SIGNAL_TEST_FILE = REPO_ROOT / "test_data" / "signal_test_manual.csv"


def run_command(args, cwd=REPO_ROOT, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print("$", " ".join(map(str, args)))
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(map(str, args))}")
    return completed


def classify_language(text: str) -> str:
    scores = language_scores(str(text))
    id_score = scores["id"] + scores["slang"]
    en_score = scores["en"]
    if id_score == 0 and en_score == 0:
        return "unclear"
    if id_score >= en_score:
        return "indonesian_dominant"
    return "english_dominant"


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_preview_frame(df: pd.DataFrame) -> pd.DataFrame:
    preview = df.copy()
    for column in preview.columns:
        if pd.api.types.is_object_dtype(preview[column]):
            preview[column] = preview[column].fillna("").astype(str).map(strip_emoji)
    return preview


def reset_social_artifacts(platform: str) -> None:
    for path in [
        REPO_ROOT / "data" / "social_media" / f"{platform}_data.csv",
        REPO_ROOT / "data" / "scraped" / "checkpoints" / f"{platform}_crawl_state.json",
    ]:
        if path.exists():
            path.unlink()
            print(f"Removed {path}")


def validate_scraping_runtime():
    scrape_python = Path(SCRAPING_PYTHON if SCRAPING_PYTHON.exists() else Path(sys.executable))
    if not scrape_python.exists():
        raise FileNotFoundError(f"Scraping Python runtime not found at {scrape_python}")
    probe = subprocess.run(
        [str(scrape_python), "-c", "import scrapling; print(scrapling.__version__)"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(probe.stderr or probe.stdout or "Failed to validate Scrapling runtime")
    print(f"Scraping runtime ready: {scrape_python}")


environment_summary = {
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu_only",
    "social_rows": json.dumps({spec["platform"]: spec["existing_rows"] for spec in SOCIAL_REFRESH_SPECS}, ensure_ascii=False),
    "social_min_rows": json.dumps({spec["platform"]: spec["min_rows"] for spec in SOCIAL_REFRESH_SPECS}, ensure_ascii=False),
    "social_refresh_plan": json.dumps([spec["platform"] for spec in SOCIAL_REFRESH_PLAN], ensure_ascii=False),
    "include_google_maps_cache": INCLUDE_GOOGLE_MAPS_CACHE,
    "run_gemini_augmentation": RUN_GEMINI_AUGMENTATION,
    "gemini_max_samples": GEMINI_MAX_SAMPLES,
    "google_maps_refresh_disabled": True,
}
display(pd.DataFrame([environment_summary]))
"""


TRAINING_EVAL_HELPERS = """
import json
from pathlib import Path

from seqeval.metrics import classification_report as seq_classification_report
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer


def evaluate_signal_checkpoint(
    model_dir: Path,
    test_path: Path,
    *,
    label_column: str = "final_signal",
    metrics_filename: str = "signal_test_metrics_pytorch.json",
    confusion_title: str = "Signal Test Confusion Matrix",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    test_df = pd.read_csv(test_path).dropna(subset=["text", label_column]).copy()
    label2id = {label: idx for idx, label in enumerate(SIGNAL_LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}

    batch_size = 16
    preds = []
    probs = []
    texts = test_df["text"].astype(str).tolist()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds.extend(batch_probs.argmax(axis=-1).tolist())
        probs.extend(batch_probs.tolist())

    true_ids = test_df[label_column].map(label2id).tolist()
    report = classification_report(
        true_ids,
        preds,
        labels=list(range(len(SIGNAL_LABELS))),
        target_names=SIGNAL_LABELS,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T

    cm = confusion_matrix(true_ids, preds, labels=list(range(len(SIGNAL_LABELS))))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=SIGNAL_LABELS, yticklabels=SIGNAL_LABELS)
    plt.title(confusion_title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    preview_rows = []
    for text, pred_id, prob in zip(texts[:10], preds[:10], probs[:10]):
        ranked = np.argsort(prob)[::-1][:3]
        preview_rows.append(
            {
                "text": text[:120],
                "top_label": id2label[pred_id],
                "top_probability": round(float(prob[pred_id]), 4),
                "runner_up": id2label[int(ranked[1])] if len(ranked) > 1 else "",
            }
        )
    preview_df = pd.DataFrame(preview_rows)

    metrics_path = REPO_ROOT / "logs" / metrics_filename
    metrics_payload = {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "samples": int(len(test_df)),
        "label_column": label_column,
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    return report_df, preview_df, metrics_payload


def load_completed_gold_signal_set(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    gold_df = pd.read_csv(path)
    if "gold_label" not in gold_df.columns:
        return pd.DataFrame()
    gold_df = gold_df.dropna(subset=["text", "gold_label"]).copy()
    gold_df["gold_label"] = gold_df["gold_label"].astype(str).str.strip()
    gold_df = gold_df[gold_df["gold_label"].isin(SIGNAL_LABELS)].copy()
    return gold_df


def predict_signal_dataframe(model_dir: Path, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    texts = df[text_column].fillna("").astype(str).tolist()
    batch_size = 16
    labels = []
    confidences = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for prob in batch_probs:
            top_index = int(np.argmax(prob))
            labels.append(SIGNAL_LABELS[top_index])
            confidences.append(round(float(prob[top_index]), 4))

    predicted = df.copy()
    predicted["final_signal"] = labels
    predicted["signal_confidence"] = confidences
    return predicted


def regex_tokens(text: str):
    return re.findall(r"\\w+|[^\\w\\s]", text, flags=re.UNICODE)


def predict_ner_tags(model, tokenizer, text: str):
    device = next(model.parameters()).device
    tokens = regex_tokens(text)
    encoded = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt")
    word_ids = encoded.word_ids(batch_index=0)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    aligned = []
    previous_word_id = None
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id == previous_word_id:
            previous_word_id = word_id
            continue
        aligned.append((tokens[word_id], model.config.id2label[pred_ids[token_index]]))
        previous_word_id = word_id
    return aligned


def extract_entities_from_tags(aligned_tags):
    entities = []
    current_tokens = []
    current_label = None
    for token, tag in aligned_tags:
        if tag.startswith("B-"):
            if current_tokens:
                entities.append({"entity": " ".join(current_tokens), "label": current_label})
            current_tokens = [token]
            current_label = tag[2:]
        elif tag.startswith("I-") and current_tokens and current_label == tag[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                entities.append({"entity": " ".join(current_tokens), "label": current_label})
            current_tokens = []
            current_label = None
    if current_tokens:
        entities.append({"entity": " ".join(current_tokens), "label": current_label})
    return entities


def evaluate_ner_checkpoint(model_dir: Path, test_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    test_data = read_json(test_path)
    true_labels = []
    pred_labels = []
    preview_rows = []

    for item in test_data:
        text = " ".join(item["tokens"])
        aligned_pred = predict_ner_tags(model, tokenizer, text)
        predicted_tags = [tag for _, tag in aligned_pred]
        min_len = min(len(item["ner_tags"]), len(predicted_tags))
        if min_len == 0:
            continue
        gold = item["ner_tags"][:min_len]
        pred = predicted_tags[:min_len]
        true_labels.append(gold)
        pred_labels.append(pred)
        if len(preview_rows) < 8:
            preview_rows.append(
                {
                    "text": text[:120],
                    "gold_entities": extract_entities_from_tags(list(zip(item["tokens"][:min_len], gold))),
                    "pred_entities": extract_entities_from_tags(list(zip(item["tokens"][:min_len], pred))),
                }
            )

    report_dict = seq_classification_report(true_labels, pred_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    metrics_path = REPO_ROOT / "logs" / "ner_test_metrics_pytorch.json"
    metrics_payload = {
        "micro_f1": float(report_dict["micro avg"]["f1-score"]),
        "samples": int(len(true_labels)),
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    preview_df = pd.DataFrame(preview_rows)
    return report_df, preview_df, metrics_payload
"""


INFERENCE_SETUP = """
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from IPython.display import display
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

REPO_ROOT = Path.cwd()
assert (REPO_ROOT / "models").exists(), "Please run this notebook from the repo root."
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SPATIAL_MODELLING_DIR = REPO_ROOT / "04_spatial_engine" / "modelling"
if str(SPATIAL_MODELLING_DIR) not in sys.path:
    sys.path.insert(0, str(SPATIAL_MODELLING_DIR))

import scoring as spatial_scoring

from common.location_resolution import LocationResolver

load_dotenv(REPO_ROOT / ".env")

SIGNAL_MODEL_DIR = REPO_ROOT / "models" / "signal_base"
NER_MODEL_DIR = REPO_ROOT / "models" / "ner_base"
READINESS_FILE = REPO_ROOT / "logs" / "production_readiness.json"
resolver = LocationResolver()
SIGNAL_LABELS = [
    "NEUTRAL",
    "DEMAND_UNMET",
    "DEMAND_PRESENT",
    "SUPPLY_SIGNAL",
    "COMPETITION_HIGH",
    "COMPLAINT",
    "TREND",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert SIGNAL_MODEL_DIR.exists(), f"Missing signal model at {SIGNAL_MODEL_DIR}"
assert NER_MODEL_DIR.exists(), f"Missing NER model at {NER_MODEL_DIR}"

signal_tokenizer = AutoTokenizer.from_pretrained(str(SIGNAL_MODEL_DIR))
signal_model = AutoModelForSequenceClassification.from_pretrained(str(SIGNAL_MODEL_DIR)).to(device)
signal_model.eval()

ner_tokenizer = AutoTokenizer.from_pretrained(str(NER_MODEL_DIR))
ner_model = AutoModelForTokenClassification.from_pretrained(str(NER_MODEL_DIR)).to(device)
ner_model.eval()

display(
    pd.DataFrame(
        [
            {
                "signal_model": str(SIGNAL_MODEL_DIR),
                "ner_model": str(NER_MODEL_DIR),
                "device": str(device),
                "readiness_file": str(READINESS_FILE if READINESS_FILE.exists() else ""),
            }
        ]
    )
)


def regex_tokens(text: str):
    return re.findall(r"\\w+|[^\\w\\s]", text, flags=re.UNICODE)


def predict_signal(texts):
    encoded = signal_tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = signal_model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    rows = []
    for text, prob in zip(texts, probs):
        ranked = np.argsort(prob)[::-1]
        rows.append(
            {
                "text": text,
                "prediction": SIGNAL_LABELS[int(ranked[0])],
                "confidence": round(float(prob[ranked[0]]), 4),
                "runner_up": SIGNAL_LABELS[int(ranked[1])],
                "runner_up_confidence": round(float(prob[ranked[1]]), 4),
                "third_choice": SIGNAL_LABELS[int(ranked[2])],
                "third_choice_confidence": round(float(prob[ranked[2]]), 4),
            }
        )
    return pd.DataFrame(rows)


def predict_ner_entities(text: str) -> list[dict[str, str]]:
    tokens = regex_tokens(text)
    encoded = ner_tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt")
    word_ids = encoded.word_ids(batch_index=0)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = ner_model(**encoded).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    aligned = []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == previous_word_id:
            previous_word_id = word_id
            continue
        aligned.append((tokens[word_id], ner_model.config.id2label[pred_ids[idx]]))
        previous_word_id = word_id

    entities = []
    current_tokens = []
    current_label = None
    for token, tag in aligned:
        if tag.startswith("B-"):
            if current_tokens:
                entities.append({"entity": " ".join(current_tokens), "label": current_label})
            current_tokens = [token]
            current_label = tag[2:]
        elif tag.startswith("I-") and current_tokens and current_label == tag[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                entities.append({"entity": " ".join(current_tokens), "label": current_label})
            current_tokens = []
            current_label = None
    if current_tokens:
        entities.append({"entity": " ".join(current_tokens), "label": current_label})
    return entities


def resolve_predicted_entities(text: str, city_hint: str = "", area_hint: str = "") -> pd.DataFrame:
    entities = predict_ner_entities(text)
    resolved = resolver.resolve_entities(entities, city_hint=city_hint, area_hint=area_hint)
    if not resolved:
        return pd.DataFrame([{"entity": "", "label": "NO_ENTITY", "resolved_city": city_hint, "resolved_area": area_hint, "lat": None, "lng": None, "resolution_source": "unresolved"}])
    return pd.DataFrame(resolved)
"""


def build_training_notebook() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # LokaSense Training Notebook

            This is the maintained end-to-end training path for the repo.
            It is public-scrape first, free-by-default, and writes the outputs directly into notebook cells.
            """
        ),
        md(
            """
            ## Notebook Flow

            1. Validate the environment and the notebook safety switches.
            2. Reuse existing public scrape data or refresh the public social sources if the local corpora are too small.
            3. Run EDA on source mix and language mix.
            4. Rebuild `signal_bootstrap.csv` and `ner_bootstrap.jsonl`.
            5. Generate weak labels and optional Gemini augmentation.
            6. Create leakage-safe train, validation, and test splits.
            7. Train and evaluate the signal model and the NER model.
            8. Run production-style opportunity scoring and write a readiness summary.
            """
        ),
        code(TRAINING_SETUP),
        md(
            """
            The setup cell shows the exact execution mode up front.
            Google Maps refresh is intentionally disabled here, and Gemini augmentation is opt-in through an environment variable instead of being silently baked into the notebook.
            """
        ),
        code(
            """
            raw_files = {
                **SOCIAL_SOURCE_FILES,
                "google_maps": REPO_ROOT / "data" / "social_media" / "gmaps_reviews.csv",
            }

            if SOCIAL_REFRESH_PLAN:
                validate_scraping_runtime()
                for spec in SOCIAL_REFRESH_PLAN:
                    if FORCE_SOCIAL_REFRESH:
                        reset_social_artifacts(str(spec["platform"]))
                    print(
                        f"Refreshing {spec['platform']} because rows={spec['existing_rows']} "
                        f"< min_rows={spec['min_rows']}"
                    )
                    run_command(build_social_refresh_args(spec))
            else:
                print("Skipping public social refresh and reusing the current raw CSV files.")

            raw_counts = []
            for source, path in raw_files.items():
                if path.exists():
                    df = pd.read_csv(path)
                    raw_counts.append({"source": source, "rows": len(df), "path": str(path)})
            raw_counts_df = pd.DataFrame(raw_counts).sort_values("rows", ascending=False)
            display(raw_counts_df)
            """
        ),
        md(
            """
            The notebook refreshes only the public social sources that are below their configured row thresholds.
            That keeps reruns practical while still giving the complaint and review-heavy platforms a chance to contribute when they are missing.
            """
        ),
        code(
            """
            eda_rows = []
            sample_frames = []

            for source, path in raw_files.items():
                if not path.exists():
                    continue

                df = pd.read_csv(path)
                text_col = "raw_text" if "raw_text" in df.columns else "text"
                texts = df[text_col].fillna("").astype(str)
                language_mix = Counter(classify_language(text) for text in texts if text.strip())

                eda_rows.append(
                    {
                        "source": source,
                        "rows": len(df),
                        "non_empty_text_rows": int((texts.str.len() > 0).sum()),
                        "avg_text_length": round(float(texts.str.len().mean()), 1),
                        "indonesian_dominant": language_mix.get("indonesian_dominant", 0),
                        "english_dominant": language_mix.get("english_dominant", 0),
                        "unclear": language_mix.get("unclear", 0),
                    }
                )

                preview_cols = [col for col in ["text", "city", "area_hint", "business_hint", "query"] if col in df.columns]
                if preview_cols:
                    preview = clean_preview_frame(df[preview_cols].head(3).copy())
                    preview.insert(0, "source", source)
                    sample_frames.append(preview)

            eda_df = pd.DataFrame(eda_rows).sort_values("rows", ascending=False)
            display(eda_df)

            plt.figure(figsize=(9, 4))
            plt.bar(eda_df["source"], eda_df["rows"], color=["#2a9d8f", "#e76f51", "#f4a261", "#457b9d"][: len(eda_df)])
            plt.title("Raw Dataset Row Counts by Source")
            plt.ylabel("Rows")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.show()

            if sample_frames:
                display(pd.concat(sample_frames, ignore_index=True))
            """
        ),
        md(
            """
            This is the first hard sanity check.
            If the Indonesian share is too small or the source mix is dominated by irrelevant text, the rest of the notebook will faithfully expose that problem instead of hiding it behind a headline metric.
            """
        ),
        code(
            """
            signal_rows = build_signal_bootstrap_rows(include_google_maps=INCLUDE_GOOGLE_MAPS_CACHE)
            signal_bootstrap_df = pd.DataFrame(signal_rows)
            signal_bootstrap_path = REPO_ROOT / "data" / "scraped" / "signal_bootstrap.csv"
            signal_bootstrap_df.to_csv(signal_bootstrap_path, index=False)

            ner_rows = build_ner_bootstrap_rows(signal_rows)
            ner_bootstrap_path = REPO_ROOT / "data" / "scraped" / "ner_bootstrap.jsonl"
            ner_bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ner_bootstrap_path, "w", encoding="utf-8") as handle:
                for row in ner_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\\n")

            print(f"signal_bootstrap_path = {signal_bootstrap_path}")
            print(f"ner_bootstrap_path = {ner_bootstrap_path}")
            print(f"signal_rows = {len(signal_bootstrap_df)}")
            print(f"ner_rows = {len(ner_rows)}")

            if not signal_bootstrap_df.empty:
                display(clean_preview_frame(signal_bootstrap_df.head(10)))
                display(
                    signal_bootstrap_df.groupby(["platform", "city"])["text"].count().rename("rows").reset_index().sort_values("rows", ascending=False)
                )

            candidate_counts = [len(row["candidate_spans"]) for row in ner_rows]
            display(
                pd.DataFrame(
                    {
                        "candidate_span_min": [int(np.min(candidate_counts)) if candidate_counts else 0],
                        "candidate_span_mean": [round(float(np.mean(candidate_counts)), 2) if candidate_counts else 0.0],
                        "candidate_span_max": [int(np.max(candidate_counts)) if candidate_counts else 0],
                    }
                )
            )
            """
        ),
        md(
            """
            The bootstrap stage is where the public scrape becomes training-ready data.
            By default it excludes Google Maps entirely, so the maintained path does not depend on a paid API or on the English-heavy review distribution that previously poisoned the signal set.
            """
        ),
        code(
            """
            run_command([sys.executable, "03_signal_model/dataset/weak_label.py"])
            weak_labeled_path = REPO_ROOT / "data" / "labeled" / "weak_labeled.csv"
            weak_df = pd.read_csv(weak_labeled_path)

            weak_counts = weak_df["signal"].value_counts(dropna=False).rename_axis("signal").reset_index(name="rows")
            weak_source_counts = weak_df["source"].value_counts(dropna=False).rename_axis("source").reset_index(name="rows")
            weak_intent_counts = weak_df["query_intent"].fillna("").replace("", "unknown").value_counts(dropna=False).rename_axis("query_intent").reset_index(name="rows") if "query_intent" in weak_df.columns else pd.DataFrame()
            display(weak_counts)
            display(weak_source_counts)
            if not weak_intent_counts.empty:
                display(weak_intent_counts)
            display(weak_df[["text", "signal", "confidence", "source", "platform", "city", "query_intent"]].head(12))
            complaint_preview = weak_df[weak_df["signal"] == "COMPLAINT"][["text", "city", "business_hint", "query", "query_intent"]].head(10)
            display(complaint_preview)
            """
        ),
        md(
            """
            Weak labels remain just a bootstrap mechanism.
            The useful part of this cell is the class coverage table, because it shows immediately whether the current scrape can support all seven business signals or whether the model is still starved on rare classes.
            The complaint preview is especially important in this project, since that class had been nearly empty before the retrieval and rule updates.
            """
        ),
        code(
            """
            gemini_file = REPO_ROOT / "data" / "labeled" / "gemini_augmented.csv"
            model_pseudo_file = REPO_ROOT / "data" / "labeled" / "model_pseudo_augmented.csv"
            if RUN_GEMINI_AUGMENTATION:
                if FORCE_GEMINI_REFRESH or not gemini_file.exists():
                    run_command(
                        [
                            sys.executable,
                            "03_signal_model/dataset/gemini_label.py",
                            "--mode",
                            "augment",
                            "--max-samples",
                            str(GEMINI_MAX_SAMPLES),
                        ]
                    )
                else:
                    print(f"Reusing existing Gemini augmentation file: {gemini_file}")
            else:
                print("Skipping Gemini augmentation. Set ENABLE_NOTEBOOK_GEMINI=1 to opt in.")

            if gemini_file.exists():
                gemini_df = pd.read_csv(gemini_file)
                gemini_counts = gemini_df["gemini_signal"].value_counts(dropna=False).rename_axis("gemini_signal").reset_index(name="rows")
                gemini_sources = gemini_df["source"].value_counts(dropna=False).rename_axis("source").reset_index(name="rows")
                display(gemini_counts)
                display(gemini_sources)
                display(gemini_df[["text", "gemini_signal", "gemini_confidence", "source", "label_source"]].head(12))
            else:
                print("No Gemini augmentation file found.")

            if RUN_MODEL_PSEUDOLABEL:
                if (REPO_ROOT / "models" / "signal_base").exists():
                    if FORCE_MODEL_PSEUDOLABEL_REFRESH or not model_pseudo_file.exists():
                        run_command(
                            [
                                sys.executable,
                                "03_signal_model/dataset/model_pseudo_label.py",
                                "--max-samples",
                                str(MODEL_PSEUDOLABEL_MAX_SAMPLES),
                            ]
                        )
                    else:
                        print(f"Reusing existing local IndoBERT pseudolabel file: {model_pseudo_file}")
                else:
                    print("Skipping local IndoBERT pseudolabeling because models/signal_base does not exist yet.")
            else:
                print("Skipping local IndoBERT pseudolabeling. Set ENABLE_NOTEBOOK_MODEL_PSEUDOLABEL=1 to opt in.")

            if model_pseudo_file.exists():
                model_pseudo_df = pd.read_csv(model_pseudo_file)
                model_pseudo_counts = model_pseudo_df["model_signal"].value_counts(dropna=False).rename_axis("model_signal").reset_index(name="rows")
                model_pseudo_sources = model_pseudo_df["source"].value_counts(dropna=False).rename_axis("source").reset_index(name="rows")
                display(model_pseudo_counts)
                display(model_pseudo_sources)
                display(model_pseudo_df[["text", "model_signal", "model_confidence", "model_margin", "source", "label_source"]].head(12))
            else:
                print("No local IndoBERT pseudolabel file found.")
            """
        ),
        md(
            """
            Gemini augmentation is now an explicit notebook switch and can contribute net-new supervised rows instead of only overriding existing labels.
            If you keep it off, the notebook still runs fully end to end using only the free public scrape path.
            If you keep it on, the notebook reuses an existing Gemini file unless you set `FORCE_NOTEBOOK_GEMINI_REFRESH=1`.
            Local IndoBERT pseudolabeling is also available for a no-API self-training pass, but it still stays train-only so the notebook does not grade the model on its own pseudolabels.
            """
        ),
        code(
            """
            run_command(
                [sys.executable, "03_signal_model/dataset/split.py"],
                extra_env={
                    "USE_GEMINI_AUGMENTATION": "1" if RUN_GEMINI_AUGMENTATION else "0",
                    "USE_GEMINI_OVERRIDES": "0",
                    "USE_MODEL_PSEUDOLABELS": "1" if RUN_MODEL_PSEUDOLABEL else "0",
                },
            )
            run_command([sys.executable, "02_ner_model/dataset/prepare.py"])

            signal_split_paths = {
                "train": REPO_ROOT / "train_data" / "signal_train.csv",
                "validation": REPO_ROOT / "train_data" / "signal_val.csv",
                "test": REPO_ROOT / "test_data" / "signal_test.csv",
            }

            signal_distribution_frames = []
            split_sizes = []
            split_text_sets = {}
            for split_name, split_path in signal_split_paths.items():
                split_df = pd.read_csv(split_path)
                split_sizes.append({"split": split_name, "rows": len(split_df)})
                split_text_sets[split_name] = set(split_df["text"].astype(str))
                counts = split_df["final_signal"].value_counts().rename(split_name)
                signal_distribution_frames.append(counts)
            signal_distribution_df = pd.concat(signal_distribution_frames, axis=1).fillna(0).astype(int)
            signal_distribution_df = signal_distribution_df.reindex(SIGNAL_LABELS, fill_value=0)
            leakage_count = len(split_text_sets["train"].intersection(split_text_sets["test"]))

            display(pd.DataFrame(split_sizes))
            display(signal_distribution_df)
            display(pd.DataFrame([{"train_test_leakage": leakage_count}]))

            with open(REPO_ROOT / "train_data" / "ner_train.json", "r", encoding="utf-8") as handle:
                ner_train = json.load(handle)
            with open(REPO_ROOT / "train_data" / "ner_val.json", "r", encoding="utf-8") as handle:
                ner_val = json.load(handle)
            with open(REPO_ROOT / "test_data" / "ner_test.json", "r", encoding="utf-8") as handle:
                ner_test = json.load(handle)

            ner_label_counts = Counter(tag for row in ner_train for tag in row["ner_tags"])
            display(pd.DataFrame([{"split": "train", "rows": len(ner_train)}, {"split": "validation", "rows": len(ner_val)}, {"split": "test", "rows": len(ner_test)}]))
            display(pd.DataFrame(sorted(ner_label_counts.items()), columns=["tag", "rows"]))
            """
        ),
        md(
            """
            This is the leakage and class-coverage checkpoint.
            If the signal split is still tiny or if rare classes disappear from validation and test, the model results later in the notebook should be treated as development diagnostics, not production evidence.
            """
        ),
        code(TRAINING_EVAL_HELPERS),
        md(
            """
            The helper cell keeps the later sections cleaner by handling direct PyTorch evaluation and notebook-native scoring.
            That avoids hiding model quality behind export-time optimizations.
            """
        ),
        code(
            """
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            run_command([sys.executable, "03_signal_model/modelling/train.py"])
            signal_metrics = read_json(REPO_ROOT / "logs" / "signal_training_metrics.json")
            display(pd.DataFrame([signal_metrics]))
            """
        ),
        md(
            """
            This cell trains the maintained signal model path and then reads the compact metrics JSON written by the trainer.
            """
        ),
        code(
            """
            signal_report_df, signal_preview_df, signal_eval_metrics = evaluate_signal_checkpoint(
                REPO_ROOT / "models" / "signal_base",
                REPO_ROOT / "test_data" / "signal_test.csv",
            )
            display(signal_report_df)
            display(signal_preview_df)
            display(pd.DataFrame([signal_eval_metrics]))
            """
        ),
        md(
            """
            The signal model is only as strong as the supervision it sees.
            This report is the honest checkpoint that should drive README claims, not any earlier experiment that used a different data regime.
            """
        ),
        md(
            """
            If a manually reviewed signal gold set exists, evaluate it separately here.
            That keeps the notebook honest about the difference between a weak-label diagnostic split and the production gate.
            """
        ),
        code(
            """
            signal_gold_metrics = None
            gold_signal_df = load_completed_gold_signal_set(MANUAL_SIGNAL_TEST_FILE)
            if gold_signal_df.empty:
                print(f"No completed manual gold signal set found at {MANUAL_SIGNAL_TEST_FILE}")
            else:
                gold_signal_path = REPO_ROOT / "test_data" / "signal_test_manual_completed.csv"
                gold_signal_df.to_csv(gold_signal_path, index=False)
                gold_report_df, gold_preview_df, signal_gold_metrics = evaluate_signal_checkpoint(
                    REPO_ROOT / "models" / "signal_base",
                    gold_signal_path,
                    label_column="gold_label",
                    metrics_filename="signal_test_metrics_gold.json",
                    confusion_title="Signal Gold Test Confusion Matrix",
                )
                display(gold_report_df)
                display(gold_preview_df)
                display(pd.DataFrame([signal_gold_metrics]))
            """
        ),
        code(
            """
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            run_command([sys.executable, "02_ner_model/modelling/train.py"])
            ner_metrics = read_json(REPO_ROOT / "logs" / "ner_training_metrics.json")
            display(pd.DataFrame([ner_metrics]))
            """
        ),
        md(
            """
            The NER path is trained after the signal model so both artifacts are refreshed inside one notebook run.
            """
        ),
        code(
            """
            ner_report_df, ner_preview_df, ner_eval_metrics = evaluate_ner_checkpoint(
                REPO_ROOT / "models" / "ner_base",
                REPO_ROOT / "test_data" / "ner_test.json",
            )
            display(ner_report_df)
            display(ner_preview_df)
            display(pd.DataFrame([ner_eval_metrics]))
            """
        ),
        md(
            """
            The NER preview is useful as a qualitative sanity check because it exposes whether the model is actually extracting entity spans instead of only posting a decent aggregate F1.
            """
        ),
        code(
            """
            scoring_input_df = predict_signal_dataframe(
                REPO_ROOT / "models" / "signal_base",
                pd.read_csv(signal_bootstrap_path),
            )
            poi_file = REPO_ROOT / "data" / "poi" / "overpass_poi.csv"
            poi_df = pd.read_csv(poi_file) if poi_file.exists() else None
            scores_df = spatial_scoring.compute_opportunity_scores(scoring_input_df, poi_df=poi_df, resolver=LocationResolver())
            if not scores_df.empty and "opportunity_score" in scores_df.columns:
                display(scores_df.sort_values("opportunity_score", ascending=False).head(20))
            else:
                print("No opportunity groups met the scoring threshold yet.")

            heatmap_output = REPO_ROOT / "outputs" / "lokasense_heatmap.html"
            marker_map = spatial_heatmap.create_marker_map(scores_df)
            marker_map.save(str(heatmap_output))
            print(f"Heatmap written to {heatmap_output}")
            """
        ),
        md(
            """
            This cell exercises the scoring path exactly the way a production batch would use it: model predictions, time-decay weighting, franchise penalties, and resolved coordinates.
            """
        ),
        code(
            """
            readiness_checks = [
                {"check": "free_by_default", "passed": True, "details": "Notebook did not run Google Maps refresh."},
                {"check": "train_test_leakage_zero", "passed": leakage_count == 0, "details": leakage_count},
                {
                    "check": "signal_train_rows",
                    "passed": int(pd.read_csv(signal_split_paths["train"]).shape[0]) >= READINESS_THRESHOLDS["signal_train_rows"],
                    "details": int(pd.read_csv(signal_split_paths["train"]).shape[0]),
                },
                {
                    "check": "signal_macro_f1",
                    "passed": float(signal_eval_metrics["macro_f1"]) >= READINESS_THRESHOLDS["signal_eval_macro_f1"],
                    "details": float(signal_eval_metrics["macro_f1"]),
                },
                {
                    "check": "signal_gold_set_present",
                    "passed": signal_gold_metrics is not None,
                    "details": int(signal_gold_metrics["samples"]) if signal_gold_metrics else 0,
                },
                {
                    "check": "signal_gold_macro_f1",
                    "passed": signal_gold_metrics is not None and float(signal_gold_metrics["macro_f1"]) >= READINESS_THRESHOLDS["signal_gold_macro_f1"],
                    "details": float(signal_gold_metrics["macro_f1"]) if signal_gold_metrics else 0.0,
                },
                {
                    "check": "ner_micro_f1",
                    "passed": float(ner_eval_metrics["micro_f1"]) >= READINESS_THRESHOLDS["ner_eval_micro_f1"],
                    "details": float(ner_eval_metrics["micro_f1"]),
                },
                {
                    "check": "opportunity_groups",
                    "passed": int(len(scores_df)) >= READINESS_THRESHOLDS["opportunity_groups"],
                    "details": int(len(scores_df)),
                },
                {
                    "check": "all_signal_classes_seen_in_train",
                    "passed": set(signal_distribution_df.index[signal_distribution_df["train"] > 0]) == set(SIGNAL_LABELS),
                    "details": int((signal_distribution_df["train"] > 0).sum()),
                },
            ]

            failed_checks = [check["check"] for check in readiness_checks if not check["passed"]]
            readiness_status = "ready" if not failed_checks else "not_ready"
            readiness_payload = {
                "status": readiness_status,
                "failed_checks": failed_checks,
                "checks": readiness_checks,
                "thresholds": READINESS_THRESHOLDS,
                "signal_metrics": signal_eval_metrics,
                "signal_gold_metrics": signal_gold_metrics,
                "ner_metrics": ner_eval_metrics,
                "opportunity_groups": int(len(scores_df)),
            }

            readiness_path = REPO_ROOT / "logs" / "production_readiness.json"
            with open(readiness_path, "w", encoding="utf-8") as handle:
                json.dump(readiness_payload, handle, indent=2)

            display(pd.DataFrame(readiness_checks))
            display(pd.DataFrame([{"production_status": readiness_status, "failed_checks": ", ".join(failed_checks)}]))
            print(f"Readiness report written to {readiness_path}")
            """
        ),
        md(
            """
            The readiness summary is the production guardrail.
            If the signal dataset is still too small or the evaluated metrics are still below threshold, the notebook should say so explicitly instead of pretending the pipeline is ready for deployment.
            """
        ),
        code(
            """
            summary_rows = [
                {"artifact": "signal_bootstrap", "path": str(REPO_ROOT / "data" / "scraped" / "signal_bootstrap.csv")},
                {"artifact": "ner_bootstrap", "path": str(REPO_ROOT / "data" / "scraped" / "ner_bootstrap.jsonl")},
                {"artifact": "signal_model", "path": str(REPO_ROOT / "models" / "signal_base")},
                {"artifact": "ner_model", "path": str(REPO_ROOT / "models" / "ner_base")},
                {"artifact": "signal_training_metrics", "path": str(REPO_ROOT / "logs" / "signal_training_metrics.json")},
                {"artifact": "ner_training_metrics", "path": str(REPO_ROOT / "logs" / "ner_training_metrics.json")},
                {"artifact": "signal_test_metrics_pytorch", "path": str(REPO_ROOT / "logs" / "signal_test_metrics_pytorch.json")},
                {"artifact": "signal_test_metrics_gold", "path": str(REPO_ROOT / "logs" / "signal_test_metrics_gold.json")},
                {"artifact": "ner_test_metrics_pytorch", "path": str(REPO_ROOT / "logs" / "ner_test_metrics_pytorch.json")},
                {"artifact": "opportunity_scores", "path": str(REPO_ROOT / "logs" / "opportunity_scores.csv")},
                {"artifact": "production_readiness", "path": str(REPO_ROOT / "logs" / "production_readiness.json")},
                {"artifact": "heatmap", "path": str(REPO_ROOT / "outputs" / "lokasense_heatmap.html")},
            ]
            display(pd.DataFrame(summary_rows))
            """
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": DISPLAY_NAME, "language": "python", "name": KERNEL_NAME},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    return nb


def build_inference_notebook() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # LokaSense Inference Notebook

            This notebook is the lightweight companion to the training notebook.
            It loads the saved checkpoints, shows the latest readiness status, runs local inference, and resolves extracted entities into usable coordinates where possible.
            """
        ),
        md(
            """
            ## What This Notebook Does

            - Loads the latest local signal and NER checkpoints
            - Displays the latest production readiness summary if one exists
            - Runs signal classification on editable Indonesian inputs
            - Extracts and resolves location-like entities
            - Shows the latest scored opportunity groups if the training notebook already generated them
            """
        ),
        code(INFERENCE_SETUP),
        md(
            """
            The setup cell fails early if a checkpoint is missing, which keeps the notebook honest about whether the training path has been run.
            """
        ),
        code(
            """
            if READINESS_FILE.exists():
                with open(READINESS_FILE, "r", encoding="utf-8") as handle:
                    readiness_payload = json.load(handle)
                display(pd.DataFrame(readiness_payload["checks"]))
                display(pd.DataFrame([{"production_status": readiness_payload["status"], "failed_checks": ", ".join(readiness_payload.get("failed_checks", []))}]))
            else:
                print("No production readiness file found yet. Run training.ipynb first.")
            """
        ),
        md(
            """
            This is the fastest way to see whether the latest notebook run actually cleared the production gates.
            """
        ),
        code(
            """
            inference_inputs = pd.DataFrame(
                [
                    {
                        "text": "tolong buka kedai kopi murah di lowokwaru malang, anak kos butuh tempat nongkrong dekat kampus",
                        "city": "Malang",
                        "area_hint": "Lowokwaru",
                        "business_hint": "kedai kopi",
                    },
                    {
                        "text": "cafe di wonokromo sekarang saingannya banyak banget, hampir tiap jalan ada tempat baru",
                        "city": "Surabaya",
                        "area_hint": "Wonokromo",
                        "business_hint": "cafe",
                    },
                    {
                        "text": "bakso ini lagi viral di kotagede jogja, rame terus dan reviewnya bagus",
                        "city": "Yogyakarta",
                        "area_hint": "Kotagede",
                        "business_hint": "bakso",
                    },
                    {
                        "text": "pelayanan warung makan di cicendo mengecewakan, mahal dan rasanya biasa aja",
                        "city": "Bandung",
                        "area_hint": "Cicendo",
                        "business_hint": "warung makan",
                    },
                ]
            )
            display(inference_inputs)
            """
        ),
        md(
            """
            Edit this cell whenever you want to test your own Indonesian text.
            The city and area hints are optional, but they help location resolution downstream.
            """
        ),
        code(
            """
            signal_predictions = predict_signal(inference_inputs["text"].tolist())
            signal_results = pd.concat([inference_inputs, signal_predictions.drop(columns=["text"])], axis=1)
            display(signal_results)
            """
        ),
        md(
            """
            The signal output exposes the top label plus the next two alternatives so uncertainty stays visible.
            """
        ),
        code(
            """
            resolved_outputs = []
            for _, row in inference_inputs.iterrows():
                resolved = resolve_predicted_entities(row["text"], city_hint=row["city"], area_hint=row["area_hint"])
                resolved_outputs.append(
                    {
                        "text": row["text"],
                        "resolved_entities": resolved.to_dict(orient="records"),
                    }
                )

            resolved_outputs_df = pd.DataFrame(resolved_outputs)
            display(resolved_outputs_df)
            """
        ),
        md(
            """
            The NER output is resolved into city, area, and coordinate hints where the gazetteer can support it.
            That closes the gap between raw entity extraction and spatial aggregation.
            """
        ),
        code(
            """
            latest_scores_path = REPO_ROOT / "logs" / "opportunity_scores.csv"
            if latest_scores_path.exists():
                latest_scores_df = pd.read_csv(latest_scores_path)
                if not latest_scores_df.empty and "opportunity_score" in latest_scores_df.columns:
                    display(latest_scores_df.sort_values("opportunity_score", ascending=False).head(20))
                else:
                    print("Opportunity scoring output exists but is still empty.")
            else:
                print("No opportunity scoring output found yet. Run training.ipynb first.")
            """
        ),
        md(
            """
            This last cell gives you the latest scored market view without retraining anything.
            """
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": DISPLAY_NAME, "language": "python", "name": KERNEL_NAME},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    return nb


def main() -> None:
    training_nb = build_training_notebook()
    inference_nb = build_inference_notebook()

    TRAINING_NOTEBOOK.write_text(nbf.writes(training_nb), encoding="utf-8")
    INFERENCE_NOTEBOOK.write_text(nbf.writes(inference_nb), encoding="utf-8")

    print(f"Wrote {TRAINING_NOTEBOOK}")
    print(f"Wrote {INFERENCE_NOTEBOOK}")


if __name__ == "__main__":
    main()

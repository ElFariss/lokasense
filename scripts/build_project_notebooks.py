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
from IPython.display import Markdown, display

REPO_ROOT = Path.cwd()
assert (REPO_ROOT / "01_data_collection").exists(), "Please run this notebook from the repo root."
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.bootstrap_utils import build_ner_bootstrap_rows, build_signal_bootstrap_rows
from common.text_normalization import language_scores, strip_emoji

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_rows", 200)
sns.set_theme(style="whitegrid")

SCRAPING_PYTHON = REPO_ROOT / ".venv_scraping" / "bin" / "python"
TIKTOK_SOURCE_FILE = REPO_ROOT / "data" / "social_media" / "tiktok_data.csv"
FORCE_TIKTOK_REFRESH = False
RUN_TIKTOK_REFRESH = FORCE_TIKTOK_REFRESH or (not TIKTOK_SOURCE_FILE.exists())
RESET_TIKTOK_SOURCE = FORCE_TIKTOK_REFRESH
TIKTOK_REFRESH_ARGS = [
    str(SCRAPING_PYTHON if SCRAPING_PYTHON.exists() else Path(sys.executable)),
    "01_data_collection/collect_social_bootstrap.py",
    "--platform", "tiktok",
    "--max-queries", "80",
    "--max-per-query", "8",
    "--max-saved-rows", "80",
    "--headless",
    "--query-delay", "0.5",
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


def reset_tiktok_artifacts():
    for path in [
        REPO_ROOT / "data" / "social_media" / "tiktok_data.csv",
        REPO_ROOT / "data" / "scraped" / "checkpoints" / "tiktok_crawl_state.json",
    ]:
        if path.exists():
            path.unlink()
            print(f"Removed {path}")


def clean_preview_frame(df: pd.DataFrame) -> pd.DataFrame:
    preview = df.copy()
    for column in preview.columns:
        if pd.api.types.is_object_dtype(preview[column]):
            preview[column] = preview[column].fillna("").astype(str).map(strip_emoji)
    return preview


def validate_scraping_runtime():
    scrape_python = Path(TIKTOK_REFRESH_ARGS[0])
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
}
display(pd.DataFrame([environment_summary]))
"""


TRAINING_EVAL_HELPERS = """
import json
from pathlib import Path

from seqeval.metrics import classification_report as seq_classification_report
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_signal_checkpoint(model_dir: Path, test_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    test_df = pd.read_csv(test_path).dropna(subset=["text", "final_signal"]).copy()
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

    true_ids = test_df["final_signal"].map(label2id).tolist()
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
    plt.title("Signal Test Confusion Matrix")
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

    metrics_path = REPO_ROOT / "logs" / "signal_test_metrics_pytorch.json"
    metrics_payload = {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "samples": int(len(test_df)),
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    return report_df, preview_df, metrics_payload


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
from IPython.display import display
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

REPO_ROOT = Path.cwd()
assert (REPO_ROOT / "models").exists(), "Please run this notebook from the repo root."
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIGNAL_MODEL_DIR = REPO_ROOT / "models" / "signal_base"
NER_MODEL_DIR = REPO_ROOT / "models" / "ner_base"
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


def predict_ner_entities(text: str):
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
    return pd.DataFrame(entities or [{"entity": "", "label": "NO_ENTITY"}])
"""


def build_training_notebook() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # LokaSense Training Notebook

            This notebook is the maintained training entry point for the project.
            It rebuilds the pipeline in notebook form so the outputs live inside the cells instead of a detached terminal session.
            It covers raw data inspection, Indonesian-first preprocessing, weak labeling, split creation, model training, and PyTorch test evaluation.
            """
        ),
        md(
            """
            ## Notebook Flow

            1. Check the training environment and available hardware.
            2. Inspect the currently available raw datasets.
            3. Run lightweight EDA on language mix, sources, and market coverage.
            4. Rebuild the scraped bootstrap datasets used by the downstream pipeline.
            5. Regenerate weak labels and train/validation/test splits.
            6. Train the signal model and the NER model.
            7. Evaluate both checkpoints directly from the saved PyTorch model folders.
            """
        ),
        code(TRAINING_SETUP),
        md(
            """
            The setup cell establishes the repo root, helper functions, and environment summary.
            If you reopen this notebook later, rerunning that cell is the quickest way to restore the working state.
            """
        ),
        code(
            """
            raw_files = {
                "tiktok": REPO_ROOT / "data" / "social_media" / "tiktok_data.csv",
                "instagram": REPO_ROOT / "data" / "social_media" / "instagram_data.csv",
                "x": REPO_ROOT / "data" / "social_media" / "x_data.csv",
                "google_maps": REPO_ROOT / "data" / "social_media" / "gmaps_reviews.csv",
            }

            if RUN_TIKTOK_REFRESH:
                validate_scraping_runtime()
                if RESET_TIKTOK_SOURCE:
                    reset_tiktok_artifacts()
                run_command(TIKTOK_REFRESH_ARGS)
            else:
                print("Skipping TikTok refresh and reusing the current raw CSV files.")

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
            This cell is the notebook-friendly replacement for the old background scraping step.
            If a TikTok source file already exists, the notebook reuses it by default so reruns stay practical.
            If you want a fresh scrape, set `FORCE_TIKTOK_REFRESH = True` in the setup cell.
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
            The EDA cell answers two practical questions before we train anything:

            - Do we actually have Indonesian-domain text available right now?
            - Which sources are dominating the pipeline?

            That matters here because the repo previously leaned too hard on English Google Maps reviews, which is a poor fit for an IndoBERT-centered classifier.
            """
        ),
        code(
            """
            signal_rows = build_signal_bootstrap_rows()
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

            display(clean_preview_frame(signal_bootstrap_df.head(10)))
            display(
                signal_bootstrap_df.groupby(["platform", "city"])["text"].count().rename("rows").reset_index().sort_values("rows", ascending=False)
            )

            candidate_counts = [
                len(row["candidate_spans"]) for row in ner_rows
            ]
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
            This preprocessing stage rebuilds the two scraped bootstrap artifacts the rest of the repo expects:

            - `signal_bootstrap.csv` for market-signal classification text
            - `ner_bootstrap.jsonl` for weak location/business spans

            The main change from the older pipeline is that the bootstrap is now filtered toward Indonesian-like text before it ever reaches training.
            """
        ),
        code(
            """
            run_command([sys.executable, "03_signal_model/dataset/weak_label.py"])
            weak_labeled_path = REPO_ROOT / "data" / "labeled" / "weak_labeled.csv"
            weak_df = pd.read_csv(weak_labeled_path)

            weak_counts = (
                weak_df["signal"]
                .value_counts(dropna=False)
                .rename_axis("signal")
                .reset_index(name="rows")
            )
            display(weak_counts)
            display(weak_df[["text", "signal", "confidence", "source"]].head(12))
            """
        ),
        md(
            """
            Weak labels are only a bootstrap layer, not the final truth.
            This cell is still worth inspecting closely because it tells you which classes the current scrape can actually support and which ones are still data-scarce.
            """
        ),
        code(
            """
            run_command([sys.executable, "03_signal_model/dataset/split.py"])
            run_command([sys.executable, "02_ner_model/dataset/prepare.py"])

            signal_split_paths = {
                "train": REPO_ROOT / "train_data" / "signal_train.csv",
                "validation": REPO_ROOT / "train_data" / "signal_val.csv",
                "test": REPO_ROOT / "test_data" / "signal_test.csv",
            }

            signal_distribution_frames = []
            for split_name, split_path in signal_split_paths.items():
                split_df = pd.read_csv(split_path)
                counts = split_df["final_signal"].value_counts().rename(split_name)
                signal_distribution_frames.append(counts)
            signal_distribution_df = pd.concat(signal_distribution_frames, axis=1).fillna(0).astype(int)
            display(signal_distribution_df)

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
            At this stage the notebook has reproduced the train/validation/test assets that the standalone scripts use.
            The signal split table is the quickest sanity check for class sparsity, while the NER tag table confirms that the label cleanup reduced the schema to the entities we actually want to model.
            """
        ),
        code(TRAINING_EVAL_HELPERS),
        md(
            """
            The helper cell above keeps the later training sections cleaner by handling PyTorch checkpoint evaluation directly in the notebook.
            That lets us show the final reports in-place without depending on the ONNX export path.
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
            The signal training cell runs the current IndoBERT trainer exactly as the repo would from the terminal.
            The metrics table below it is the compact summary you can scan without digging through the raw trainer log.
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
            The notebook uses direct PyTorch evaluation here on purpose.
            It keeps the training notebook focused on model quality first, before any pruning or ONNX optimization choices enter the picture.
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
            This NER training stage should now produce a much cleaner label space than the earlier mixed-schema runs.
            If the label count suddenly jumps again, that usually means the raw NER tags are leaking inconsistent source labels back into preprocessing.
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
            The NER evaluation preview is especially useful for spot-checking whether the model is finding real place-like spans in Indonesian text instead of just memorizing a news-domain tag distribution.
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
                {"artifact": "ner_test_metrics_pytorch", "path": str(REPO_ROOT / "logs" / "ner_test_metrics_pytorch.json")},
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
            Train or refresh checkpoints from `training.ipynb`, then use this notebook to run local inference directly from cells.
            It loads the saved signal and NER checkpoints and runs end-user inference directly from notebook cells.
            """
        ),
        md(
            """
            ## What This Notebook Does

            - Loads the latest local `models/signal_base` checkpoint
            - Loads the latest local `models/ner_base` checkpoint
            - Runs signal classification on editable Indonesian example text
            - Extracts NER spans from the same text so you can quickly inspect location and organization hits
            """
        ),
        code(INFERENCE_SETUP),
        md(
            """
            The setup cell loads both models into memory and binds them to the active device.
            If a checkpoint is missing, it will fail early with a clear path so you know which training artifact still needs to be generated.
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
            Edit the `inference_inputs` cell if you want to test your own Indonesian text.
            Keeping `city`, `area_hint`, and `business_hint` nearby also makes it easier to reuse these rows later for downstream scoring or demos.
            """
        ),
        code(
            """
            signal_predictions = predict_signal(inference_inputs["text"].tolist())
            display(pd.concat([inference_inputs, signal_predictions.drop(columns=["text"])], axis=1))
            """
        ),
        md(
            """
            The signal classifier output shows the top prediction and the next two alternatives.
            That is usually more useful than a single hard label when you are debugging borderline texts like `DEMAND_UNMET` versus `DEMAND_PRESENT`.
            """
        ),
        code(
            """
            ner_outputs = []
            for _, row in inference_inputs.iterrows():
                entities = predict_ner_entities(row["text"]).to_dict(orient="records")
                ner_outputs.append({"text": row["text"], "entities": entities})

            ner_outputs_df = pd.DataFrame(ner_outputs)
            display(ner_outputs_df)
            """
        ),
        md(
            """
            The NER output is deliberately kept simple here: you get back the extracted entities and their labels per text.
            That makes the notebook easy to use as a local demo surface without needing to run the full spatial aggregation pipeline.
            """
        ),
        code(
            """
            batch_demo_path = REPO_ROOT / "data" / "social_media" / "tiktok_data.csv"
            if batch_demo_path.exists():
                batch_demo = pd.read_csv(batch_demo_path).head(8).copy()
                batch_demo_predictions = predict_signal(batch_demo["text"].fillna("").astype(str).tolist())
                display(pd.concat([batch_demo[["text", "city", "area_hint", "business_hint"]], batch_demo_predictions.drop(columns=["text"])], axis=1))
            else:
                print(f"No demo batch found at {batch_demo_path}")
            """
        ),
        md(
            """
            This optional batch cell gives you a quick sanity check on real scraped text without retraining anything.
            It is a nice first stop when you want to know whether the saved checkpoints are at least behaving plausibly on the current raw corpus.
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

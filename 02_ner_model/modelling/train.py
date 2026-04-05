#!/usr/bin/env python3
"""
02_ner_model/modelling/train.py
Fine-tune indobert-base-p1 for NER (token classification).
"""
import os, json, time
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)
from seqeval.metrics import f1_score as seq_f1

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = Path(__file__).parent.parent.parent
TRAIN_DIR = BASE_DIR / "train_data"
LOG_DIR = BASE_DIR / "logs"
MODEL_OUT = BASE_DIR / "models" / "ner_base"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
EVAL_ACCUMULATION_STEPS = 4


def load_ner_data(filepath):
    with open(filepath) as f:
        return json.load(f)


def build_label_list(data):
    labels = set()
    for s in data:
        labels.update(s["ner_tags"])
    return sorted(labels)


def align_labels(tokenized_inputs, labels, tokenizer, label2id):
    """Align labels with WordPiece tokenization."""
    aligned = []
    for i, label_list in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(label2id.get(label_list[word_id], 0))
            else:
                # Subword: use I- version or -100
                lbl = label_list[word_id]
                if lbl.startswith("B-"):
                    aligned_labels.append(label2id.get("I-" + lbl[2:], label2id.get(lbl, 0)))
                else:
                    aligned_labels.append(label2id.get(lbl, 0))
            prev_word_id = word_id
        aligned.append(aligned_labels)
    return aligned


def main():
    print("=" * 60)
    print(" NER Model Training — LokaSense")
    print("=" * 60)

    train_data = load_ner_data(TRAIN_DIR / "ner_train.json")
    val_data = load_ner_data(TRAIN_DIR / "ner_val.json")
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

    # Get unique labels
    all_data = train_data + val_data
    label_list = build_label_list(all_data)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    print(f"  Labels ({len(label_list)}): {label_list}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize and align
    def tokenize_and_align(data_list):
        texts = [s["tokens"] for s in data_list]
        labels = [s["ner_tags"] for s in data_list]
        tokenized = tokenizer(texts, is_split_into_words=True, truncation=True, max_length=MAX_LENGTH, return_tensors=None)
        aligned = align_labels(tokenized, labels, tokenizer, label2id)
        tokenized["labels"] = aligned
        return Dataset.from_dict(tokenized)

    train_ds = tokenize_and_align(train_data)
    val_ds = tokenize_and_align(val_data)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if isinstance(predictions, list):
            predictions = predictions[0]
        preds = np.argmax(predictions, axis=-1) if getattr(predictions, "ndim", 1) > 2 else predictions
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_preds = [[id2label[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
        f1 = seq_f1(true_labels, true_preds, average="macro")
        return {"ner_f1": f1}

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.argmax(logits, dim=-1)

    args = TrainingArguments(
        output_dir=str(BASE_DIR / "results_ner"),
        overwrite_output_dir=True,
        learning_rate=5e-5,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
        num_train_epochs=8,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="ner_f1",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        group_by_length=True,
        warmup_ratio=0.1,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))

    # Save label mapping
    with open(MODEL_OUT / "label_mapping.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    metrics = {
        "training_time_sec": elapsed,
        "train_loss": result.training_loss,
        "best_metric": trainer.state.best_metric,
        "label_count": len(label_list),
        "labels": label_list,
    }
    with open(LOG_DIR / "ner_training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nNER training complete in {elapsed:.0f}s")
    print(f"  Best F1: {trainer.state.best_metric:.4f}")
    print(f"  Model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
03_signal_model/modelling/train.py
Fine-tune indobert-base-p1 for 7-class market signal classification.
Saves metrics to logs/ at each epoch.
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = Path(__file__).parent.parent.parent
TRAIN_DIR = BASE_DIR / "train_data"
LOG_DIR = BASE_DIR / "logs"
MODEL_OUT = BASE_DIR / "models" / "signal_base"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
EVAL_ACCUMULATION_STEPS = 8
SIGNAL_LABELS = ["NEUTRAL", "DEMAND_UNMET", "DEMAND_PRESENT", "SUPPLY_SIGNAL", "COMPETITION_HIGH", "COMPLAINT", "TREND"]
label2id = {l: i for i, l in enumerate(SIGNAL_LABELS)}
id2label = {i: l for i, l in enumerate(SIGNAL_LABELS)}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self._class_weights.to(outputs.logits.device))
        loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(predictions, list):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=-1) if getattr(predictions, "ndim", 1) > 1 else predictions
    label_range = np.arange(len(SIGNAL_LABELS))
    macro_f1 = f1_score(labels, preds, labels=label_range, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, labels=label_range, average=None, zero_division=0)
    return {"macro_f1": macro_f1, **{f"f1_{SIGNAL_LABELS[i]}": float(per_class[i]) for i in range(len(SIGNAL_LABELS))}}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def main():
    print("=" * 60)
    print(" Signal Classifier Training — LokaSense")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv(TRAIN_DIR / "signal_train.csv")
    val_df = pd.read_csv(TRAIN_DIR / "signal_val.csv")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_df['final_signal'] = train_df['final_signal'].map(label2id)
    val_df['final_signal'] = val_df['final_signal'].map(label2id)
    
    # Check for unmapped labels
    train_df = train_df.dropna(subset=['final_signal'])
    val_df = val_df.dropna(subset=['final_signal'])
    train_df['final_signal'] = train_df['final_signal'].astype(int)
    val_df['final_signal'] = val_df['final_signal'].astype(int)

    # Datasets
    train_ds = Dataset.from_pandas(train_df[['text', 'final_signal', 'source']])
    val_ds = Dataset.from_pandas(val_df[['text', 'final_signal', 'source']])

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    train_tok = train_ds.map(tokenize, batched=True).rename_column("final_signal", "labels")
    val_tok = val_ds.map(tokenize, batched=True).rename_column("final_signal", "labels")

    # Class weights
    labels_arr = train_df['final_signal'].values
    present_classes = np.unique(labels_arr)
    present_weights = compute_class_weight("balanced", classes=present_classes, y=labels_arr)
    weight_map = {int(label): float(weight) for label, weight in zip(present_classes, present_weights)}
    cw = np.array([weight_map.get(index, 1.0) for index in range(len(SIGNAL_LABELS))], dtype=np.float32)
    class_weights = torch.tensor(cw, dtype=torch.float)
    print(f"  Class weights: {dict(zip(SIGNAL_LABELS, [f'{w:.2f}' for w in cw]))}")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(SIGNAL_LABELS), id2label=id2label, label2id=label2id)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir=str(BASE_DIR / "results_signal"),
        overwrite_output_dir=True,
        learning_rate=3e-5,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
        num_train_epochs=7,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        group_by_length=True,
        warmup_ratio=0.1,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Train
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    # Save model
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))

    # Save training metrics
    metrics = {
        "training_time_sec": elapsed,
        "train_loss": result.training_loss,
        "best_metric": trainer.state.best_metric,
        "epochs_trained": result.metrics.get("epoch", 5),
        "model_name": MODEL_NAME,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }
    
    # Eval history
    eval_history = []
    for log in trainer.state.log_history:
        if "eval_macro_f1" in log:
            eval_history.append(log)
    metrics["eval_history"] = eval_history

    with open(LOG_DIR / "signal_training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"  Best macro F1: {trainer.state.best_metric:.4f}")
    print(f"  Model saved to: {MODEL_OUT}")
    print(f"  Metrics saved to: logs/signal_training_metrics.json")


if __name__ == "__main__":
    main()

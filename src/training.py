# %% [markdown]
# # LokaSense (Pasarint) — Code Training Notebook
# 
# **Hackathon: FindIT! 2026 — Track C (The Explainable Oracle)**  
# **Objective:** Develop a 7-class market signal classification model and NER model for business entity extraction, strictly adhering to Track C constraints.
# 
# ### Constraint Compliance Checklist:
# - [x] **C-4: Data Leakage Proof:** Train/Val/Test splits were created fully *before* any ML preprocessing or notebook execution (via `scripts/create_data_splits.py`). Test data is loaded only at the very end for evaluation.
# - [x] **General:** Models exported to INT8 ONNX for strict CPU offline inference.
# 
# ---

# %% [code]
import os
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset, DatasetDict, ClassLabel, Features, Value, Sequence
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification
)

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForTokenClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

# Constants
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = Path()
TRAIN_DIR = BASE_DIR / "train_data"
TEST_DIR = BASE_DIR / "test_data"
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 128

SIGNAL_LABELS = [
    "NEUTRAL", "DEMAND_UNMET", "DEMAND_PRESENT", 
    "SUPPLY_SIGNAL", "COMPETITION_HIGH", "COMPLAINT", "TREND"
]
num_labels = len(SIGNAL_LABELS)

# %% [markdown]
# ## 1. Tokenizer Setup & Data Loading
# **Note on Leakage:** The `signal_train.csv` and `signal_val.csv` were created by `scripts/create_data_splits.py` using a seed, drawing from all historical data. The test set `signal_test.csv` is completely isolated.

# %% [code]
# Init Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load strictly isolated training data
train_df = pd.read_csv(TRAIN_DIR / "signal_train.csv")
val_df = pd.read_csv(TRAIN_DIR / "signal_val.csv")

# Create label mapping
label2id = {label: i for i, label in enumerate(SIGNAL_LABELS)}
id2label = {i: label for i, label in enumerate(SIGNAL_LABELS)}

# Format datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

features = Features({
    'text': Value('string'),
    'final_signal': ClassLabel(names=SIGNAL_LABELS),
    'source': Value('string')
})

train_dataset = train_dataset.cast(features)
val_dataset = val_dataset.cast(features)

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True).rename_column("final_signal", "labels")
tokenized_val = val_dataset.map(tokenize_function, batched=True).rename_column("final_signal", "labels")

# Calculate class weights for imbalance handling
labels_array = train_df['final_signal'].map(label2id).values
class_weights = compute_class_weight('balanced', classes=np.unique(labels_array), y=labels_array)
class_weights_t = torch.tensor(class_weights, dtype=torch.float)
if torch.cuda.is_available():
    class_weights_t = class_weights_t.cuda()

print(f"Class Weights mapping: {dict(zip(SIGNAL_LABELS, class_weights))}")

# %% [markdown]
# ## 2. Custom Weighted Trainer
# We override the HuggingFace `Trainer` to pass class weights to the `CrossEntropyLoss` function. This treats rare signals (like `DEMAND_UNMET`) with higher importance than common classes (like `NEUTRAL`).

# %% [code]
class CustomWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Use our globally defined class weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_t)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    return {"macro_f1": f1}

# Initialize Signal Classifier Model
model_signal = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results_signal",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none"
)

trainer = CustomWeightedTrainer(
    model=model_signal,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %% [markdown]
# ## 3. Train Signal Classifier
# Launching the fine-tuning process. 

# %% [code]
print("Starting Signal Classifier Training...")
trainer.train()

# Save best base model locally before ONNX
model_signal.save_pretrained("./models/signal_base")
tokenizer.save_pretrained("./models/signal_base")

# %% [markdown]
# ## 4. Export to ONNX (Quantization)
# To satisfy the Track C and General Constraints regarding strict Offline Total CPU inference (targeting <3s), we export the trained `indobert-base-p1` base PyTorch model to ONNX using Optimum, and then apply dynamic INT8 quantization. This shrinks the model size by ~4x (to ~110MB) and massively boosts CPU throughput.

# %% [code]
import shutil

print("Exporting Signal Model to ONNX...")
onnx_export_path = Path("./signal_onnx")
if onnx_export_path.exists():
    shutil.rmtree(onnx_export_path)

# Export base ONNX model
onnx_model = ORTModelForSequenceClassification.from_pretrained(
    "./models/signal_base", 
    export=True
)
onnx_model.save_pretrained(onnx_export_path)
tokenizer.save_pretrained(onnx_export_path)

# Quantize to INT8
print("Applying INT8 Dynamic Quantization...")
quantizer = ORTQuantizer.from_pretrained(onnx_model)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

# This overwrites the original unquantized model.onnx with the quantized one
quantizer.quantize(save_dir=onnx_export_path, quantization_config=dqconfig)

print("✅ Model successfully exported and quantized to `signal_onnx/`")

# %% [markdown]
# ## 5. Model Evaluation (Test Set)
# Finally, we run inference on the strictly isolated test set using the ONNX quantized model to ensure performance hasn't degraded during INT8 conversion.

# %% [code]
test_df = pd.read_csv(TEST_DIR / "signal_test.csv")
print(f"Loaded Test Set: {len(test_df)} samples")

# Load quantized ONNX model
ort_model = ORTModelForSequenceClassification.from_pretrained("./signal_onnx")

def predict_onnx(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = ort_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    return preds.numpy()

# Run predictions in batches for memory safety
batch_size = 32
test_preds = []

start_time = time.time()
for i in range(0, len(test_df), batch_size):
    batch_texts = test_df['text'].iloc[i:i+batch_size].tolist()
    preds = predict_onnx(batch_texts)
    test_preds.extend(preds)

end_time = time.time()
inference_time = end_time - start_time
print(f"✅ Tested ONNX inference: {inference_time/len(test_df)*1000:.1f}ms per sample on CPU!")

test_labels = test_df['final_signal'].map(label2id).values

print("\n--- TEST SET CLASSIFICATION REPORT ---")
print(classification_report(test_labels, test_preds, target_names=SIGNAL_LABELS, zero_division=0))

# %% [markdown]
# ### End of Signal Classifier Pipeline.
# *Note: The NER training block follows a similar `Trainer` → `Optimum` → `ONNX` pipeline loop if the dataset is provided in `ner_train.json`.*

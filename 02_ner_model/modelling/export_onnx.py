#!/usr/bin/env python3
"""
02_ner_model/modelling/export_onnx.py
Export NER model to ONNX + INT8 dynamic quantization.
"""
import shutil, json

from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_IN = BASE_DIR / "models" / "ner_base"
ONNX_OUT = BASE_DIR / "ner_onnx"
LOG_DIR = BASE_DIR / "logs"

def main():
    print("=" * 60)
    print(" ONNX Export + INT8 Quantization — NER Model")
    print("=" * 60)

    if not MODEL_IN.exists():
        print(f"Model not found at {MODEL_IN}. Run training.ipynb first.")
        return

    import torch
    import torch.nn.utils.prune as prune
    from transformers import AutoModelForTokenClassification

    print("  Applying post-training magnitude pruning before ONNX export...")
    pt_model = AutoModelForTokenClassification.from_pretrained(str(MODEL_IN))
    
    # Apply 20% magnitude pruning to all Linear layers
    for name, module in pt_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prune.remove(module, 'weight')
            
    pruned_dir = BASE_DIR / "models" / "ner_pruned"
    pt_model.save_pretrained(str(pruned_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_IN))
    tokenizer.save_pretrained(str(pruned_dir))

    print("  Exporting pruned model to ONNX...")
    ort_model = ORTModelForTokenClassification.from_pretrained(str(pruned_dir), export=True)
    ort_model.save_pretrained(str(ONNX_OUT))
    tokenizer.save_pretrained(str(ONNX_OUT))

    onnx_file = ONNX_OUT / "model.onnx"
    unquant_size_mb = onnx_file.stat().st_size / (1024 * 1024) if onnx_file.exists() else 0

    print("  Applying INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=str(ONNX_OUT), quantization_config=dqconfig)

    quant_file = ONNX_OUT / "model_quantized.onnx"
    if not quant_file.exists():
        quant_file = onnx_file
    quant_size_mb = quant_file.stat().st_size / (1024 * 1024)

    metrics = {
        "unquantized_size_mb": round(unquant_size_mb, 1),
        "quantized_size_mb": round(quant_size_mb, 1),
        "compression_ratio": round(unquant_size_mb / max(quant_size_mb, 1), 2),
        "quantization": "INT8_dynamic_avx2",
    }
    with open(LOG_DIR / "ner_onnx_export.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nExport complete.")
    print(f"  Unquantized: {unquant_size_mb:.1f} MB → Quantized: {quant_size_mb:.1f} MB")
    print(f"  Saved to: {ONNX_OUT}")


if __name__ == "__main__":
    main()

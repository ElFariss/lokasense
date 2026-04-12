#!/usr/bin/env python3
"""
03_signal_model/modelling/export_onnx.py
Export signal classifier to ONNX + INT8 dynamic quantization.
"""
import shutil, json
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_IN = BASE_DIR / "models" / "signal_base"
ONNX_OUT = BASE_DIR / "signal_onnx"
LOG_DIR = BASE_DIR / "logs"


def main():
    print("=" * 60)
    print(" ONNX Export + INT8 Quantization — Signal Classifier")
    print("=" * 60)

    if not MODEL_IN.exists():
        print(f"Model not found at {MODEL_IN}. Run training.ipynb first.")
        return

    import torch
    import torch.nn.utils.prune as prune
    from transformers import AutoModelForSequenceClassification

    print("  Applying post-training magnitude pruning before ONNX export...")
    pt_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_IN))
    
    # Apply 20% magnitude pruning to all Linear layers
    for name, module in pt_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prune.remove(module, 'weight')
            
    pruned_dir = BASE_DIR / "models" / "signal_pruned"
    pt_model.save_pretrained(str(pruned_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_IN))
    tokenizer.save_pretrained(str(pruned_dir))

    print("  Exporting pruned model to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(str(pruned_dir), export=True)
    ort_model.save_pretrained(str(ONNX_OUT))
    tokenizer.save_pretrained(str(ONNX_OUT))

    # Calculate unquantized size
    onnx_file = ONNX_OUT / "model.onnx"
    unquant_size_mb = onnx_file.stat().st_size / (1024 * 1024) if onnx_file.exists() else 0

    # INT8 dynamic quantization
    print("  Applying INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=str(ONNX_OUT), quantization_config=dqconfig)

    # Calculate quantized size
    quant_file = ONNX_OUT / "model_quantized.onnx"
    if not quant_file.exists():
        quant_file = onnx_file  # avx512 may overwrite in place
    quant_size_mb = quant_file.stat().st_size / (1024 * 1024)

    # Save export metrics
    metrics = {
        "unquantized_size_mb": round(unquant_size_mb, 1),
        "quantized_size_mb": round(quant_size_mb, 1),
        "compression_ratio": round(unquant_size_mb / max(quant_size_mb, 1), 2),
        "quantization": "INT8_dynamic_avx2",
    }
    with open(LOG_DIR / "signal_onnx_export.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nExport complete.")
    print(f"  Unquantized: {unquant_size_mb:.1f} MB → Quantized: {quant_size_mb:.1f} MB")
    print(f"  Saved to: {ONNX_OUT}")


if __name__ == "__main__":
    main()

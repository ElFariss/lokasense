#!/usr/bin/env python3
"""
Gemini-powered pseudo-labeling for 7-class market signal classification.
Uses Gemini 2.5 Flash to label texts that weak labeling couldn't classify
with high confidence, or as a refinement layer on top of weak labels.

This is a DATA COLLECTION/PREPROCESSING step — NOT part of inference.
Gemini API is used only here, never in the inference pipeline.

Usage:
    python scripts/gemini_pseudo_label.py [--mode refine|label_all|low_confidence]
"""
import os
import json
import csv
import time
import argparse
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
ENV_DIR = BASE_DIR.parent
load_dotenv(ENV_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
LABELED_DIR = DATA_DIR / "labeled"
OUTPUT_FILE = LABELED_DIR / "gemini_labeled.csv"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── Signal Class Definitions ──────────────────────────────────────────────

SIGNAL_DEFINITIONS = """
Kamu adalah ahli analisis sentimen pasar Indonesia yang SANGAT tepat.
Tugas: klasifikasikan teks ulasan/posting publik ke SATU dari 7 sinyal pasar berikut:

1. DEMAND_UNMET — Permintaan yang belum terpenuhi. Seseorang mengeluh tidak ada, belum tersedia, atau ingin tetapi tidak bisa mendapatkannya di lokasi tersebut.
   Contoh: "Di Malang belum ada selat solo yang enak", "Pengen ayam geprek tapi ga ada di sini"

2. DEMAND_PRESENT — Permintaan yang sudah ada dan terpenuhi dengan baik. Ulasan positif tentang pengalaman yang memuaskan.
   Contoh: "Bakso di sini enak banget, murah lagi!", "Wajib coba kopi di tempat ini"

3. SUPPLY_SIGNAL — Observasi FAKTUAL tentang jumlah/ketersediaan bisnis serupa. Bukan opini, tapi fakta.
   Contoh: "Udah ada 3 outlet Mixue di Lowokwaru", "Cabang baru mie gacoan buka di Blimbing"

4. COMPETITION_HIGH — Persepsi SUBJEKTIF bahwa persaingan terlalu tinggi/jenuh.
   Contoh: "Cafe di sini dimana-mana, banyak banget", "Udah kebanyakan yang jualan ayam geprek"

5. COMPLAINT — Keluhan tentang kualitas, harga, atau layanan. Peluang diferensiasi.
   Contoh: "Mahal banget harganya ga worth it", "Pelayanan lambat, tempatnya kotor"

6. TREND — Sinyal viral, trending, atau timing. Sesuatu yang sedang hype.
   Contoh: "Lagi viral di TikTok!", "FYP mulu cafe ini, harus coba"

7. NEUTRAL — Tidak mengandung sinyal pasar yang jelas. Informasi umum.
   Contoh: "Tempat ini buka jam 8 pagi", "Alamatnya di jalan Diponegoro"

PENTING:
- Jawab HANYA dengan format JSON: {"signal": "SIGNAL_NAME", "confidence": 0.0-1.0, "reason": "alasan singkat"}
- Confidence 0.9-1.0 = sangat yakin, 0.7-0.9 = cukup yakin, <0.7 = kurang yakin
- Jika teks ambigu, berikan confidence rendah
- SUPPLY_SIGNAL vs COMPETITION_HIGH: SUPPLY = fakta (ada 3 toko), COMPETITION = opini (terlalu banyak)
"""

FEW_SHOT_EXAMPLES = [
    {"text": "Di Malang belum ada dimsum yang enak kayak di Jakarta", "signal": "DEMAND_UNMET", "confidence": 0.95, "reason": "Ekspresi kebutuhan yang belum terpenuhi di lokasi spesifik"},
    {"text": "Bakso Pak Min emang enaaak bgt, murah meriah mantap", "signal": "DEMAND_PRESENT", "confidence": 0.92, "reason": "Ulasan sangat positif tentang makanan yang sudah ada"},
    {"text": "Udah ada 3 outlet Mixue sekarang di Lowokwaru", "signal": "SUPPLY_SIGNAL", "confidence": 0.90, "reason": "Observasi faktual tentang jumlah outlet"},
    {"text": "Banyak banget yang jualan ayam geprek di sini, dimana-mana", "signal": "COMPETITION_HIGH", "confidence": 0.88, "reason": "Persepsi subjektif bahwa persaingan terlalu tinggi"},
    {"text": "Mahal banget, porsi kecil, ga worth it sih", "signal": "COMPLAINT", "confidence": 0.93, "reason": "Keluhan tentang harga dan porsi"},
    {"text": "Lagi viral di TikTok cafe baru ini, FYP mulu!", "signal": "TREND", "confidence": 0.91, "reason": "Sinyal viral/trending di media sosial"},
    {"text": "Buka dari jam 8 pagi sampai jam 10 malam", "signal": "NEUTRAL", "confidence": 0.95, "reason": "Informasi faktual tanpa sinyal pasar"},
]


def init_gemini():
    """Initialize Gemini client."""
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file!")

    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-04-17",
        generation_config={
            "temperature": 0.1,
            "top_p": 0.8,
            "max_output_tokens": 200,
            "response_mime_type": "application/json",
        },
    )
    return model


def classify_with_gemini(model, text: str) -> dict:
    """Classify a single text using Gemini."""
    # Build few-shot examples string
    examples_str = "\n".join([
        f'Teks: "{ex["text"]}"\nJawab: {{"signal": "{ex["signal"]}", "confidence": {ex["confidence"]}, "reason": "{ex["reason"]}"}}'
        for ex in FEW_SHOT_EXAMPLES
    ])

    prompt = f"""{SIGNAL_DEFINITIONS}

Contoh:
{examples_str}

Sekarang klasifikasikan teks berikut:
Teks: "{text}"
Jawab:"""

    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return {
            "signal": result.get("signal", "NEUTRAL"),
            "confidence": float(result.get("confidence", 0.5)),
            "reason": result.get("reason", ""),
        }
    except json.JSONDecodeError:
        # Try to extract from non-JSON response
        text_response = response.text if hasattr(response, 'text') else str(response)
        for signal in ["DEMAND_UNMET", "DEMAND_PRESENT", "SUPPLY_SIGNAL", "COMPETITION_HIGH", "COMPLAINT", "TREND", "NEUTRAL"]:
            if signal in text_response:
                return {"signal": signal, "confidence": 0.5, "reason": "parsed from non-JSON"}
        return {"signal": "NEUTRAL", "confidence": 0.3, "reason": "failed to parse"}
    except Exception as e:
        print(f"  ⚠ Gemini error: {e}")
        return {"signal": "NEUTRAL", "confidence": 0.0, "reason": f"error: {str(e)[:50]}"}


def classify_batch(model, texts: list, batch_size: int = 5) -> list:
    """Classify texts in batches with rate limiting."""
    results = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        batch_results = []

        for text_item in batch:
            result = classify_with_gemini(model, text_item["text"])
            batch_results.append({
                **text_item,
                "gemini_signal": result["signal"],
                "gemini_confidence": result["confidence"],
                "gemini_reason": result["reason"],
            })

            # Rate limiting: ~10 requests per minute for free tier
            time.sleep(0.5)

        results.extend(batch_results)

        # Progress
        done = min(i + batch_size, total)
        print(f"  [{done}/{total}] Processed {done} texts ({done / total * 100:.0f}%)")

        # Checkpoint save every 100 texts
        if done % 100 == 0:
            _save_checkpoint(results)

        # Rate limit pause between batches
        time.sleep(1.0)

    return results


def _save_checkpoint(results):
    """Save checkpoint."""
    if not results:
        return
    checkpoint_file = LABELED_DIR / "gemini_labeled_checkpoint.csv"
    fieldnames = results[0].keys()
    with open(checkpoint_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Gemini pseudo-labeling for market signals")
    parser.add_argument("--mode", choices=["refine", "label_all", "low_confidence"],
                        default="low_confidence",
                        help="refine: relabel all with Gemini; label_all: label everything; low_confidence: only low-conf weak labels")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum samples to process (for rate limit management)")
    args = parser.parse_args()

    print("=" * 70)
    print(" Gemini Pseudo-Labeling — 7-Class Market Signal")
    print(f" Mode: {args.mode} | Max samples: {args.max_samples}")
    print("=" * 70)

    # Load weak-labeled data
    weak_labeled_file = LABELED_DIR / "weak_labeled.csv"
    if not weak_labeled_file.exists():
        print("\n⚠ Run weak_labeling.py first!")
        return

    texts_to_label = []
    with open(weak_labeled_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.mode == "low_confidence":
                # Only label texts where weak labeling confidence is low
                conf = float(row.get("confidence", 0))
                if conf < 0.8 or row.get("signal", "") == "NEUTRAL":
                    texts_to_label.append(row)
            elif args.mode == "refine":
                # Relabel everything (use Gemini as ground truth)
                texts_to_label.append(row)
            else:  # label_all
                texts_to_label.append(row)

    # Limit samples
    if len(texts_to_label) > args.max_samples:
        import random
        random.seed(42)
        texts_to_label = random.sample(texts_to_label, args.max_samples)

    print(f"\n  Texts to label: {len(texts_to_label)}")

    if not texts_to_label:
        print("  Nothing to label!")
        return

    # Initialize Gemini
    print("\n🤖 Initializing Gemini 2.5 Flash...")
    model = init_gemini()

    # Classify
    print("\n🏷️  Running pseudo-labeling...")
    results = classify_batch(model, texts_to_label)

    # Save results
    if results:
        fieldnames = results[0].keys()
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Statistics
    signal_counts = Counter(r["gemini_signal"] for r in results)
    print(f"\n📊 Gemini Signal Distribution:")
    for signal, count in signal_counts.most_common():
        pct = count / len(results) * 100
        print(f"  {signal:20s}  {count:>5}  ({pct:5.1f}%)")

    high_conf = sum(1 for r in results if r["gemini_confidence"] >= 0.8)
    print(f"\n  High confidence (≥0.8): {high_conf} ({high_conf / len(results) * 100:.1f}%)")

    # Compare with weak labels (if mode=refine)
    if args.mode == "refine":
        agree = sum(1 for r in results if r.get("signal") == r.get("gemini_signal"))
        print(f"  Agreement with weak labels: {agree}/{len(results)} ({agree / len(results) * 100:.1f}%)")

    print(f"\n✅ Saved {len(results)} Gemini-labeled texts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

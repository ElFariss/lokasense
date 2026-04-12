# LokaSense

LokaSense is an Indonesian-first market signal and location extraction pipeline for local business opportunity mapping.

The maintained workflow is notebook-first:

- run [training.ipynb](training.ipynb) for EDA, preprocessing, weak labeling, optional Gemini augmentation, training, evaluation, scoring, and readiness checks,
- run [inference.ipynb](inference.ipynb) for local inference and resolved entity inspection.
- run `python analyze.py "<query>"` for the live query-to-heatmap runtime path.

Dataset structure, provenance, and regeneration notes are documented in [DATA.md](DATA.md).

## Current Scope

What is real in this repo now:

- Public social scraping with Scrapling plus Playwright-backed browser control.
- Free-by-default data collection path using TikTok, Instagram, X, and OpenStreetMap Overpass.
- Optional Google Maps enrichment, but only behind explicit billable confirmation.
- IndoBERT-based signal classification and NER training.
- Live `analyze.py` flow that scrapes public data, scores opportunities, and writes an interactive heatmap.
- Time-decayed opportunity scoring with franchise penalties and resolved coordinates.
- Notebook-written training outputs, evaluation reports, scoring outputs, and a production readiness report.

What is not claimed anymore:

- No knowledge distillation claim.
- No WikiAnn augmentation claim.
- No stale `0.93` signal macro F1 claim.
- No AVX-512-only deployment claim.

## Latest Notebook Run

Latest executed notebook run in this workspace on April 7, 2026:

- Signal train rows: `1360`
- Signal validation rows: `121`
- Signal test rows: `143`
- Signal best validation macro F1: `0.7160`
- Signal test macro F1: `0.7109`
- Signal test accuracy: `0.7902`
- NER test micro F1: `0.8048`
- Opportunity groups scored: `24`
- Production readiness: `ready`

This run used `255` train-only IndoBERT self-training pseudolabels and did not call any paid or quota-limited API.
The repo still supports optional Gemini augmentation, but the maintained no-API path now uses the local IndoBERT pseudolabel file when enabled.

These values come from:

- `logs/signal_training_metrics.json`
- `logs/signal_test_metrics_pytorch.json`
- `logs/ner_test_metrics_pytorch.json`
- `logs/production_readiness.json`

## Cost Safety

The default maintained path does not call billable APIs.

- Google Maps refresh is disabled in the notebooks and requires `--confirm-billable` if run manually.
- Gemini augmentation is optional and must be explicitly enabled for notebook execution.
- Existing local files can still be reused without triggering network cost.

## Main Artifacts

- `data/scraped/signal_bootstrap.csv`
- `data/scraped/ner_bootstrap.jsonl`
- `models/signal_base`
- `models/ner_base`
- `logs/signal_training_metrics.json`
- `logs/signal_test_metrics_pytorch.json`
- `logs/ner_training_metrics.json`
- `logs/ner_test_metrics_pytorch.json`
- `logs/opportunity_scores.csv`
- `logs/production_readiness.json`
- `outputs/lokasense_heatmap.html`

## Training

Build or rebuild the notebooks:

```bash
.venv/bin/python scripts/build_project_notebooks.py
```

Execute the maintained training path:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace training.ipynb --ExecutePreprocessor.kernel_name=ugm_hackathon --ExecutePreprocessor.timeout=-1
```

Optional Gemini augmentation during notebook execution:

```bash
ENABLE_NOTEBOOK_GEMINI=1 NOTEBOOK_GEMINI_MAX_SAMPLES=1200 .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace training.ipynb --ExecutePreprocessor.kernel_name=ugm_hackathon --ExecutePreprocessor.timeout=-1
```

Run inference after training:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace inference.ipynb --ExecutePreprocessor.kernel_name=ugm_hackathon --ExecutePreprocessor.timeout=-1
```

## Production Readiness

The training notebook writes `logs/production_readiness.json` and marks the current run as either:

- `ready`
- `not_ready`

That status is based on leakage checks, dataset size, evaluated signal and NER metrics, and whether the scoring stage produced a usable set of grouped opportunities.

Use that readiness file instead of guessing from a single metric or from an outdated README table.
For a final production-ready claim, complete the manual gold review flow with `scripts/build_manual_test_set.py` and rerun `training.ipynb` so the gold gate is reflected in `logs/production_readiness.json`.

The offline serving gate is written separately to `logs/airgap_production_readiness.json`.
That file distinguishes between:

- `operational_status`: the airgapped runtime works with the local corpus and passes latency plus acceptance checks.
- `status`: the stricter production claim, which also requires a manually reviewed gold signal set instead of AI-assisted review metadata.

## Runtime Analysis

Run the default airgapped runtime entry point:

```bash
.venv/bin/python analyze.py "saya ingin memulai bisnis kedai kopi di Malang"
```

This default path is offline-first:

- it uses the local `data/airgap/airgap_corpus.csv`,
- it does not call billable services,
- and it writes run-local artifacts under `outputs/airgap_runs/`.

Runtime profiles:

- `--profile full`
  Most accurate local path in this repo. Uses the full runtime stack and is best for laptops.
- `--profile edge`
  Smaller local path for weaker laptops. Prefers the quantized signal model, disables LIME, and skips NER.
- `--profile mobile`
  Smallest local path. Uses the quantized signal model, disables LIME, and relies on resolver-based location extraction instead of the NER model.

Example mobile-style local run:

```bash
.venv/bin/python analyze.py --profile mobile "saya ingin memulai bisnis kedai kopi di Malang"
```

If you explicitly want the public live scraping path instead, use:

```bash
.venv/bin/python analyze.py --live "saya ingin memulai bisnis kedai kopi di Malang"
```

That path scrapes public sources at analysis time and writes run-local artifacts under `outputs/live_runs/`.

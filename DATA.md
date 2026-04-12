# Data Guide

This repository mixes three kinds of data:

1. Small static reference datasets that are safe to keep in Git.
2. Third-party benchmark datasets vendored into the repo for reproducibility.
3. Generated local artifacts such as scraped rows, weak labels, pseudolabels, splits, logs, models, and outputs.

The third category is intentionally ignored by `.gitignore` because those files can grow quickly, may change per run, and are better treated as build artifacts than source code. The codebase is written so those artifacts can be regenerated through the maintained notebook workflow.

## Layout

### `data/geospatial/`

Administrative boundary reference data used for location normalization and spatial scoring.

- `data/geospatial/Wilayah-Administratif-Indonesia/`
  Indonesian administrative tables and upstream metadata.
- Used by:
  - `common/location_resolution.py`
  - `04_spatial_engine/modelling/scoring.py`
  - `04_spatial_engine/modelling/heatmap.py`

### `data/poi/`

Point-of-interest support files used by the gazetteer matcher and location candidate extraction.

- Example:
  - `data/poi/overpass_poi.csv`
- Used by:
  - `common/bootstrap_utils.py`
  - `common/location_resolution.py`

### `data/slang/`

Normalization resources for Indonesian informal and social-media text.

- Example:
  - `data/slang/slang_dict.json`
- Used by:
  - `common/text_normalization.py`

### `data/indolem_ner/`

Vendored IndoLEM NER data and helper code used for the maintained NER training path.

- Includes:
  - UGM and UI fold files such as `train.01.tsv`, `dev.01.tsv`, and `test.01.tsv`
- Used by:
  - `02_ner_model/dataset/prepare.py`
  - `03_signal_model/dataset/split.py`

### `data/nusax_sentiment/`

Vendored NusaX sentiment data.

- In the current maintained pipeline this is not used as a supervised headline benchmark.
- It may still be mined as auxiliary Indonesian text for weak-labeling and pseudolabel candidate generation.

### `data/huggingface/`

Vendored text-classification datasets used only as auxiliary raw Indonesian text sources.

- `data/huggingface/indonesian_sentiment/`
  Indonesian review/sentiment text used as candidate text for complaint-heavy weak labeling and pseudolabel expansion.
- `data/huggingface/smsa/`
  Legacy folder from older experiments.
- `data/huggingface/nerp/`
  Legacy folder from older experiments.

Important:

- The current maintained workflow does not present these datasets as final task evaluation.
- They are treated as text reservoirs, not as authoritative market-signal labels.

## Generated Local Artifacts

The following directories usually exist locally after running the project, but they are treated as generated artifacts:

### `data/social_media/`

Public-source collection outputs.

Typical files:

- `data/social_media/tiktok_data.csv`
- `data/social_media/instagram_data.csv`
- `data/social_media/x_data.csv`
- `data/social_media/gmaps_reviews.csv`

What they contain:

- raw or normalized post/review text
- source URL
- timestamps
- city and area hints
- business hints
- lightweight scrape metadata

How they are produced:

- `01_data_collection/collect_social_bootstrap.py`
- `01_data_collection/social_bootstrap.py`
- optionally `01_data_collection/collect_gmaps_reviews.py`

Cost note:

- The maintained path is free-by-default.
- Google Maps collection is optional and explicitly billable only when manually enabled.

### `data/scraped/`

Bootstrapped training-ready intermediate files derived from social/raw text.

Typical files:

- `data/scraped/signal_bootstrap.csv`
- `data/scraped/ner_bootstrap.jsonl`
- `data/scraped/manifest.json`

What they contain:

- classifier-ready signal text rows
- weak NER candidate spans
- provenance and run summaries

How they are produced:

- `common/bootstrap_utils.py`
- `01_data_collection/social_bootstrap.py`
- `training.ipynb`

### `data/labeled/`

Generated labeling artifacts.

Typical files:

- `data/labeled/weak_labeled.csv`
- `data/labeled/weak_label_summary.json`
- `data/labeled/gemini_augmented.csv`
- `data/labeled/gemini_augmented_checkpoint.csv`
- `data/labeled/model_pseudo_augmented.csv`

Meaning:

- `weak_labeled.csv`
  Rule-based bootstrap labels used as the baseline supervised pool.
- `gemini_augmented.csv`
  Optional Gemini-generated pseudolabel rows.
- `model_pseudo_augmented.csv`
  Local IndoBERT self-training pseudolabel rows.

Current maintained policy:

- Pseudolabel rows are train-only.
- Validation and test remain on the weak-labeled split produced by `03_signal_model/dataset/split.py`.

## Other Generated Directories Outside `data/`

These are also build artifacts rather than source data:

- `train_data/`
  Final train and validation splits.
- `test_data/`
  Final held-out test splits.
- `models/`
  Trained checkpoints.
- `logs/`
  Training metrics, evaluation metrics, readiness reports, and scored outputs.
- `outputs/`
  Visualization artifacts such as the HTML heatmap.

## What The Maintained Training Notebook Actually Uses

The notebook-first workflow in `training.ipynb` currently relies on:

1. Public social scrape outputs when available.
2. Geospatial and POI reference tables.
3. IndoLEM NER data.
4. Auxiliary Indonesian sentiment/review text for weak-labeling and pseudolabel mining.
5. Train-only pseudolabel expansion from either:
   - Gemini, if explicitly enabled, or
   - the local IndoBERT self-training path.

## Reproducibility Expectations

To regenerate the data artifacts used by the maintained pipeline:

1. Rebuild notebooks:
   `python scripts/build_project_notebooks.py`
2. Run the training notebook:
   `python -m jupyter nbconvert --to notebook --execute --inplace training.ipynb --ExecutePreprocessor.kernel_name=ugm_hackathon --ExecutePreprocessor.timeout=-1`
3. Run the inference notebook:
   `python -m jupyter nbconvert --to notebook --execute --inplace inference.ipynb --ExecutePreprocessor.kernel_name=ugm_hackathon --ExecutePreprocessor.timeout=-1`

Optional toggles:

- `ENABLE_NOTEBOOK_GEMINI=1`
  Use Gemini pseudolabels.
- `ENABLE_NOTEBOOK_MODEL_PSEUDOLABEL=1`
  Use the local IndoBERT self-training pseudolabel path.
- `FORCE_NOTEBOOK_SOCIAL_REFRESH=1`
  Re-scrape the public social sources.

## What Is Safe To Assume On GitHub

If a file is committed in this repo, treat it as:

- a static reference asset,
- an upstream benchmark snapshot,
- or source code.

If a file is produced by training, scraping, pseudolabeling, model export, or scoring, assume it is a local artifact unless the README explicitly says otherwise.

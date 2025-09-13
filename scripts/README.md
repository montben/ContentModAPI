# Muzzle ML Pipeline Scripts

This directory contains the complete ML pipeline for training the Muzzle content moderation models.

## Directory Structure

```
scripts/
├── data_collection/     # Download and collect datasets
│   ├── download_jigsaw.py
│   ├── download_hateval.py
│   ├── download_davidson.py
│   └── collect_all.py   # Master script
├── preprocessing/       # Data cleaning and label mapping
│   ├── label_schema.py  # 8-label definition
│   ├── merge_datasets.py
│   └── create_splits.py
├── training/           # Model training
│   ├── train_bert.py   # Main training script
│   └── config.py       # Training hyperparameters
└── evaluation/         # Model evaluation and optimization
    ├── evaluate_model.py
    ├── threshold_sweep.py
    └── generate_metrics.py
```

## Pipeline Execution Order

1. **Data Collection**: `python scripts/data_collection/collect_all.py`
2. **Preprocessing**: `python scripts/preprocessing/merge_datasets.py`
3. **Training**: `python scripts/training/train_bert.py`
4. **Evaluation**: `python scripts/evaluation/threshold_sweep.py`

## 8-Label Schema

- `toxicity` - General toxic language
- `hate_speech` - Targeted hate based on identity
- `harassment` - Personal attacks and bullying
- `self_harm` - Suicide ideation and self-injury
- `violence` - Threats and violent content
- `sexual` - Sexual harassment and inappropriate content
- `profanity` - Explicit language and swearing
- `spam` - Promotional content and off-topic posts

Each text sample gets an 8-bit vector: `[0,1,0,1,0,0,1,0]` indicating which labels apply.

## Usage

```bash
# Full pipeline
./run_pipeline.sh

# Individual steps
python scripts/data_collection/collect_all.py
python scripts/preprocessing/merge_datasets.py
python scripts/training/train_bert.py
python scripts/evaluation/threshold_sweep.py
```

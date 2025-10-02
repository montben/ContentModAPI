"""
Download Jigsaw Civil Comments dataset for content moderation.

This is a comprehensive dataset that covers multiple toxicity types:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_attack

Dataset: google/civil_comments (300k+ labeled comments)
This single dataset replaces the need for multiple smaller datasets.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from datasets import load_dataset

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocessing.label_schema import create_label_vector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def map_civil_comments_labels(row) -> dict:
    """
    Map Civil Comments dataset labels to our 8-label schema.

    Civil Comments provides toxicity scores (0-1) which we threshold at 0.5.
    """
    threshold = 0.5

    # Get toxicity indicators
    toxic = row.get('toxicity', 0) >= threshold
    severe_toxic = row.get('severe_toxicity', 0) >= threshold
    obscene = row.get('obscene', 0) >= threshold
    threat = row.get('threat', 0) >= threshold
    insult = row.get('insult', 0) >= threshold
    identity_attack = row.get('identity_attack', 0) >= threshold
    sexual_explicit = row.get('sexual_explicit', 0) >= threshold

    return {
        "toxicity": toxic or severe_toxic,
        "hate_speech": identity_attack,  # Identity-based attacks
        "harassment": insult or (toxic and identity_attack),  # Personal attacks
        "violence": threat or severe_toxic,  # Threats indicate violence
        "sexual": obscene or sexual_explicit,  # Sexual/obscene content
        "profanity": obscene,  # Profane language
    }


def download_civil_comments(output_dir: str, sample_size: Optional[int] = None) -> Path:
    """
    Download and process Jigsaw Civil Comments dataset.

    Args:
        output_dir: Directory to save the dataset
        sample_size: Optional limit on number of samples (for testing)

    Returns:
        Path to the downloaded dataset directory
    """
    output_path = Path(output_dir) / "civil_comments"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Civil Comments dataset to {output_path}")

    try:
        # Load Civil Comments dataset from HuggingFace
        logger.info("Loading civil_comments dataset from HuggingFace Hub...")
        logger.info("This may take a few minutes for the first download...")

        # Load train split (it's the largest and most comprehensive)
        dataset = load_dataset("google/civil_comments", split="train")
        logger.info(f"✅ Successfully loaded {len(dataset)} samples")

        # Convert to pandas DataFrame
        df = dataset.to_pandas()

        # Sample data if requested (useful for testing)
        if sample_size and sample_size < len(df):
            logger.info(f"Sampling {sample_size} examples for testing")
            # Stratified sampling to keep balance of toxic vs non-toxic
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # Process the data
        processed_data = []

        logger.info("Converting dataset labels to 8-label schema...")

        for idx, row in df.iterrows():
            # Convert dataset labels to our 8-label schema
            labels_dict = map_civil_comments_labels(row)

            # Create label vector
            label_vector = create_label_vector(labels_dict)

            # Create processed row
            processed_row = {
                'text': row['text'],
                'source': 'civil_comments',
                'original_id': row.get('id', f'sample_{idx}'),
                # Add individual label columns
                **labels_dict,
                # Add label vector as string for CSV storage
                'label_vector': ','.join(map(str, label_vector))
            }

            processed_data.append(processed_row)

            # Progress logging
            if (idx + 1) % 5000 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} samples...")

        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)

        # Save processed data
        output_file = output_path / "processed_data.csv"
        processed_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")

        # Save raw data as well (for reference) - just a sample to save space
        raw_sample = df.head(1000) if len(df) > 1000 else df
        raw_file = output_path / "raw_data.csv"
        raw_sample.to_csv(raw_file, index=False)
        logger.info(f"Saved raw data sample to {raw_file}")

        # Create summary statistics
        create_dataset_summary(processed_df, output_path)

        logger.info(f"✅ Civil Comments dataset successfully downloaded and processed!")
        logger.info(f"📊 Total samples: {len(processed_df):,}")
        logger.info(f"📁 Output directory: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"❌ Failed to download Civil Comments dataset: {str(e)}")
        logger.error(f"Error details: {type(e).__name__}")
        raise


def create_dataset_summary(df: pd.DataFrame, output_path: Path):
    """Create a summary of the dataset statistics."""

    # Label columns (our 6-label schema)
    label_columns = ['toxicity', 'hate_speech', 'harassment',
                    'violence', 'sexual', 'profanity']

    # Calculate statistics
    stats = {
        'total_samples': len(df),
        'label_distribution': {}
    }

    for label in label_columns:
        if label in df.columns:
            count = df[label].sum()
            percentage = (count / len(df)) * 100
            stats['label_distribution'][label] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }

    # Text length statistics
    df['text_length'] = df['text'].str.len()
    stats['text_statistics'] = {
        'avg_length': round(df['text_length'].mean(), 2),
        'min_length': int(df['text_length'].min()),
        'max_length': int(df['text_length'].max()),
        'median_length': int(df['text_length'].median())
    }

    # Multi-label statistics
    label_cols_mask = df[label_columns].astype(bool)
    labels_per_sample = label_cols_mask.sum(axis=1)
    stats['multi_label_stats'] = {
        'avg_labels_per_sample': round(labels_per_sample.mean(), 2),
        'samples_with_no_labels': int((labels_per_sample == 0).sum()),
        'samples_with_multiple_labels': int((labels_per_sample > 1).sum())
    }

    # Save summary
    summary_file = output_path / "dataset_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("# Civil Comments Dataset Summary\n\n")
        f.write(f"Total samples: {stats['total_samples']:,}\n\n")
        f.write("## Label Distribution:\n")
        for label, info in stats['label_distribution'].items():
            f.write(f"- {label}: {info['count']:,} ({info['percentage']}%)\n")
        f.write(f"\n## Text Statistics:\n")
        f.write(f"- Average length: {stats['text_statistics']['avg_length']} characters\n")
        f.write(f"- Median length: {stats['text_statistics']['median_length']} characters\n")
        f.write(f"- Min length: {stats['text_statistics']['min_length']} characters\n")
        f.write(f"- Max length: {stats['text_statistics']['max_length']} characters\n")
        f.write(f"\n## Multi-Label Statistics:\n")
        f.write(f"- Average labels per sample: {stats['multi_label_stats']['avg_labels_per_sample']}\n")
        f.write(f"- Samples with no labels: {stats['multi_label_stats']['samples_with_no_labels']:,}\n")
        f.write(f"- Samples with multiple labels: {stats['multi_label_stats']['samples_with_multiple_labels']:,}\n")

    logger.info(f"📈 Dataset summary saved to {summary_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Civil Comments dataset")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for dataset")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size for testing (default: use all ~300k samples)")

    args = parser.parse_args()

    try:
        download_civil_comments(args.output_dir, args.sample_size)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


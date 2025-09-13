"""
Download Davidson et al. Hate Speech Dataset.

This script downloads the Davidson hate speech dataset from GitHub and processes it
for use in the Muzzle content moderation pipeline.
"""

import os
import sys
import logging
import pandas as pd
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocessing.label_schema import map_davidson_labels, create_label_vector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset URL
DAVIDSON_URL = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"


def download_davidson(output_dir: str, sample_size: Optional[int] = None) -> Path:
    """
    Download and process Davidson et al. hate speech dataset.

    Args:
        output_dir: Directory to save the dataset
        sample_size: Optional limit on number of samples (for testing)

    Returns:
        Path to the downloaded dataset directory
    """
    output_path = Path(output_dir) / "davidson"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Davidson dataset to {output_path}")

    try:
        # Download dataset from GitHub
        logger.info(f"Downloading from: {DAVIDSON_URL}")
        response = requests.get(DAVIDSON_URL)
        response.raise_for_status()

        # Save raw data
        raw_file = output_path / "raw_data.csv"
        with open(raw_file, 'wb') as f:
            f.write(response.content)
        logger.info(f"Raw data saved to {raw_file}")

        # Load and process data
        df = pd.read_csv(raw_file)
        logger.info(f"Loaded {len(df)} training samples")

        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampling {sample_size} examples for testing")

        # Convert dataset labels to our 8-label schema
        logger.info("Converting dataset labels to 8-label schema...")
        logger.info(f"Dataset columns: {list(df.columns)}")

        processed_data = []

        for idx, row in df.iterrows():
            # Convert dataset labels to our 8-label schema
            labels_dict = map_davidson_labels(row)

            # Create label vector
            label_vector = create_label_vector(labels_dict)

            # Create processed row
            processed_row = {
                'text': row['tweet'],
                'source': 'davidson',
                'original_id': f'davidson_{idx}',
                # Add individual label columns
                **labels_dict,
                # Add label vector as string for CSV storage
                'label_vector': ','.join(map(str, label_vector))
            }

            processed_data.append(processed_row)

            # Progress logging
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} samples...")

        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)

        # Save processed data
        processed_file = output_path / "processed_data.csv"
        processed_df.to_csv(processed_file, index=False)
        logger.info(f"Saved processed data to {processed_file}")

        # Create summary statistics
        create_dataset_summary(processed_df, output_path)

        logger.info(f"✅ Davidson dataset successfully downloaded and processed!")
        logger.info(f"📊 Total samples: {len(processed_df)}")
        logger.info(f"📁 Output directory: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"❌ Failed to download Davidson dataset: {str(e)}")
        raise


def create_dataset_summary(df: pd.DataFrame, output_path: Path):
    """Create summary statistics for the dataset."""
    summary_file = output_path / "dataset_summary.txt"

    # Calculate label statistics
    label_columns = ['toxicity', 'hate_speech', 'harassment', 'self_harm',
                    'violence', 'sexual', 'profanity', 'spam']

    with open(summary_file, 'w') as f:
        f.write(f"# Davidson Dataset Summary\n\n")
        f.write(f"Total samples: {len(df):,}\n\n")

        f.write("## Label Distribution:\n")
        for label in label_columns:
            count = df[label].sum()
            percentage = (count / len(df)) * 100
            f.write(f"- {label}: {count:,} ({percentage:.2f}%)\n")

        f.write("\n## Text Statistics:\n")
        text_lengths = df['text'].str.len()
        f.write(f"- Average length: {text_lengths.mean():.2f} characters\n")
        f.write(f"- Median length: {text_lengths.median():.0f} characters\n")
        f.write(f"- Min length: {text_lengths.min()} characters\n")
        f.write(f"- Max length: {text_lengths.max()} characters\n")

    logger.info(f"📈 Dataset summary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Davidson hate speech dataset")
    parser.add_argument("--sample-size", type=int, help="Limit number of samples for testing")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for datasets")

    args = parser.parse_args()

    download_davidson(args.output_dir, args.sample_size)

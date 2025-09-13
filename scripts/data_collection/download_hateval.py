"""
Download HatEval dataset.

This script downloads the HatEval dataset from HuggingFace Hub and processes it
for use in the Muzzle content moderation pipeline.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset
from huggingface_hub import login
import argparse
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocessing.label_schema import map_hateval_labels, create_label_vector

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authenticate with HuggingFace if token is available
hf_token = os.getenv('HUGGING_FACE_TOKEN')
if hf_token:
    try:
        login(token=hf_token)
        logger.info("✅ Successfully authenticated with HuggingFace")
    except Exception as e:
        logger.warning(f"⚠️ Failed to authenticate with HuggingFace: {e}")
else:
    logger.warning("⚠️ No HUGGING_FACE_TOKEN found in environment variables")


def download_hateval(output_dir: str, sample_size: Optional[int] = None) -> Path:
    """
    Download and process HatEval dataset.

    Args:
        output_dir: Directory to save the dataset
        sample_size: Optional limit on number of samples (for testing)

    Returns:
        Path to the downloaded dataset directory
    """
    output_path = Path(output_dir) / "hateval"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading HatEval dataset to {output_path}")

    try:
        # Load HatEval dataset from HuggingFace Hub
        logger.info("Loading HatEval dataset from HuggingFace Hub...")

        try:
            dataset = load_dataset("valeriobasile/HatEval")
            logger.info("✅ Successfully loaded HatEval dataset")
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise

        # Process training data (combine English and Spanish if available)
        train_data = dataset['train']
        logger.info(f"Loaded {len(train_data)} training samples")

        # Convert to pandas DataFrame
        df = train_data.to_pandas()

        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampling {sample_size} examples for testing")

        # Convert dataset labels to our 8-label schema
        logger.info("Converting dataset labels to 8-label schema...")
        logger.info(f"Dataset columns: {list(df.columns)}")

        # Find text column (could be 'text' or 'tweet')
        text_column = None
        for col in ['text', 'tweet', 'content']:
            if col in df.columns:
                text_column = col
                break

        if text_column is None:
            raise ValueError(f"No text column found. Available columns: {list(df.columns)}")

        processed_data = []

        for idx, row in df.iterrows():
            # Convert dataset labels to our 8-label schema
            labels_dict = map_hateval_labels(row)

            # Create label vector
            label_vector = create_label_vector(labels_dict)

            # Create processed row
            processed_row = {
                'text': row[text_column],
                'source': 'hateval',
                'original_id': row.get('id', f'hateval_{idx}'),
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

        # Save raw data
        raw_file = output_path / "raw_data.csv"
        df.to_csv(raw_file, index=False)
        logger.info(f"Saved raw data to {raw_file}")

        # Create summary statistics
        create_dataset_summary(processed_df, output_path)

        logger.info(f"✅ HatEval dataset successfully downloaded and processed!")
        logger.info(f"📊 Total samples: {len(processed_df)}")
        logger.info(f"📁 Output directory: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"❌ Failed to download HatEval dataset: {str(e)}")
        raise


def create_sample_data(output_path: Path, num_samples: int) -> Path:
    """Create sample HatEval data for testing if real dataset unavailable."""
    logger.info(f"Creating {num_samples} sample HatEval entries...")

    sample_data = []
    for i in range(num_samples):
        # Create balanced sample data
        is_hate = i % 3 == 0  # ~33% hate speech
        is_aggressive = i % 4 == 0  # ~25% aggressive
        is_individual = i % 2 == 0  # ~50% individual targets

        sample_row = {
            'text': f"Sample hate speech text {i}" if is_hate else f"Sample normal text {i}",
            'source': 'hateval_sample',
            'original_id': f'sample_{i}',
            'toxicity': is_hate or is_aggressive,
            'hate_speech': is_hate,
            'harassment': is_hate and is_individual,
            'self_harm': False,
            'violence': is_aggressive,
            'sexual': False,
            'profanity': is_aggressive,
            'spam': False,
            'label_vector': ','.join(map(str, create_label_vector({
                'toxicity': is_hate or is_aggressive,
                'hate_speech': is_hate,
                'harassment': is_hate and is_individual,
                'self_harm': False,
                'violence': is_aggressive,
                'sexual': False,
                'profanity': is_aggressive,
                'spam': False
            })))
        }
        sample_data.append(sample_row)

    df = pd.DataFrame(sample_data)
    processed_file = output_path / "processed_data.csv"
    df.to_csv(processed_file, index=False)

    create_dataset_summary(df, output_path)
    logger.info(f"✅ Sample HatEval data created: {len(df)} samples")
    return output_path


def create_dataset_summary(df: pd.DataFrame, output_path: Path):
    """Create summary statistics for the dataset."""
    summary_file = output_path / "dataset_summary.txt"

    # Calculate label statistics
    label_columns = ['toxicity', 'hate_speech', 'harassment', 'self_harm',
                    'violence', 'sexual', 'profanity', 'spam']

    with open(summary_file, 'w') as f:
        f.write(f"# HatEval Dataset Summary\n\n")
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
    parser = argparse.ArgumentParser(description="Download HatEval dataset")
    parser.add_argument("--sample-size", type=int, help="Limit number of samples for testing")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for datasets")

    args = parser.parse_args()

    download_hateval(args.output_dir, args.sample_size)
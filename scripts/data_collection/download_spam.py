"""
Download SMS Spam dataset.

This script downloads SMS spam datasets from HuggingFace Hub and processes them
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

from scripts.preprocessing.label_schema import create_label_vector

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


def map_spam_labels(spam_row: Dict) -> Dict[str, bool]:
    """
    Map spam dataset labels to our 8-label schema.

    Assumes spam datasets have a binary label (spam/ham or 1/0)
    """
    # Try different column names for spam labels
    is_spam = False
    for spam_col in ['label', 'spam', 'is_spam', 'class']:
        if spam_col in spam_row:
            label_val = spam_row[spam_col]
            # Handle different spam label formats
            if isinstance(label_val, str):
                is_spam = label_val.lower() in ['spam', '1', 'true']
            else:
                is_spam = bool(label_val)
            break

    return {
        "toxicity": False,      # Spam is not necessarily toxic
        "hate_speech": False,
        "harassment": False,
        "self_harm": False,
        "violence": False,
        "sexual": False,
        "profanity": False,
        "spam": is_spam         # Only spam label is relevant
    }


def download_spam(output_dir: str, sample_size: Optional[int] = None) -> Path:
    """
    Download and process spam detection dataset.

    Args:
        output_dir: Directory to save the dataset
        sample_size: Optional limit on number of samples (for testing)

    Returns:
        Path to the downloaded dataset directory
    """
    output_path = Path(output_dir) / "spam"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading spam dataset to {output_path}")

    # Try multiple spam datasets
    dataset_options = [
        "sms_spam",
        "Deysi/spam-detection-dataset",
        "uciml/sms-spam-collection-dataset",
        "spam_ham_dataset"
    ]

    dataset = None
    dataset_name = None

    for ds_name in dataset_options:
        try:
            logger.info(f"Trying to load dataset: {ds_name}")
            dataset = load_dataset(ds_name)
            dataset_name = ds_name
            logger.info(f"✅ Successfully loaded {ds_name}")
            break
        except Exception as e:
            logger.warning(f"❌ Failed to load {ds_name}: {e}")
            continue

    if dataset is None:
        logger.error("❌ No spam datasets could be loaded")
        raise Exception("No available spam datasets found")

    # Process training data
    train_data = dataset['train'] if 'train' in dataset else list(dataset.values())[0]
    logger.info(f"Loaded {len(train_data)} training samples from {dataset_name}")

    # Convert to pandas DataFrame
    df = train_data.to_pandas()

    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampling {sample_size} examples for testing")

    # Convert dataset labels to our 8-label schema
    logger.info("Converting dataset labels to 8-label schema...")
    logger.info(f"Dataset columns: {list(df.columns)}")

    # Find text column
    text_column = None
    for col in ['text', 'message', 'sms', 'content', 'email']:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        raise ValueError(f"No text column found. Available columns: {list(df.columns)}")

    processed_data = []

    for idx, row in df.iterrows():
        # Convert dataset labels to our 8-label schema
        labels_dict = map_spam_labels(row)

        # Create label vector
        label_vector = create_label_vector(labels_dict)

        # Create processed row
        processed_row = {
            'text': row[text_column],
            'source': f'spam_{dataset_name.replace("/", "_")}',
            'original_id': row.get('id', f'spam_{idx}'),
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

    logger.info(f"✅ Spam dataset successfully downloaded and processed!")
    logger.info(f"📊 Total samples: {len(processed_df)}")
    logger.info(f"📁 Output directory: {output_path}")

    return output_path


def create_dataset_summary(df: pd.DataFrame, output_path: Path):
    """Create summary statistics for the dataset."""
    summary_file = output_path / "dataset_summary.txt"

    # Calculate label statistics
    label_columns = ['toxicity', 'hate_speech', 'harassment', 'self_harm',
                    'violence', 'sexual', 'profanity', 'spam']

    with open(summary_file, 'w') as f:
        f.write(f"# Spam Dataset Summary\n\n")
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
    parser = argparse.ArgumentParser(description="Download spam detection dataset")
    parser.add_argument("--sample-size", type=int, help="Limit number of samples for testing")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for datasets")

    args = parser.parse_args()

    download_spam(args.output_dir, args.sample_size)

"""
Download sexually explicit comments dataset from Kaggle.

This script downloads the sexually explicit comments dataset from Kaggle
and processes it for use in the Muzzle content moderation pipeline.
"""

import os
import sys
import logging
import pandas as pd
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
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


def map_sexual_labels(row: Dict) -> Dict[str, bool]:
    """
    Map sexual content dataset labels to our 8-label schema.

    Assumes the dataset has labels indicating sexual/explicit content.
    """
    # Try different possible column names for sexual content labels
    is_sexual = False

    # Check various possible label columns
    for col in ['label', 'is_sexual', 'explicit', 'sexual', 'class']:
        if col in row:
            label_val = row[col]
            if isinstance(label_val, str):
                # String labels like "sexual", "explicit", "1"
                is_sexual = label_val.lower() in ['sexual', 'explicit', 'yes', '1', 'true']
            else:
                # Numeric labels (1 = sexual, 0 = not sexual)
                is_sexual = bool(label_val)
            break

    # If no explicit label column, assume all data is sexual (since it's from a sexual content dataset)
    if not any(col in row for col in ['label', 'is_sexual', 'explicit', 'sexual', 'class']):
        is_sexual = True

    return {
        "toxicity": is_sexual,      # Sexual content can be considered toxic
        "hate_speech": False,       # Not necessarily hate speech
        "harassment": is_sexual,    # Sexual content can be harassment
        "self_harm": False,
        "violence": False,
        "sexual": is_sexual,        # Primary label
        "profanity": is_sexual,     # Sexual content often includes profanity
        "spam": False
    }


def download_sexual_content(output_dir: str, sample_size: Optional[int] = None) -> Path:
    """
    Download and process sexual content dataset from Kaggle.

    Args:
        output_dir: Directory to save the dataset
        sample_size: Optional limit on number of samples (for testing)

    Returns:
        Path to the downloaded dataset directory
    """
    output_path = Path(output_dir) / "sexual_content"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading sexual content dataset to {output_path}")

    try:
        # Import kaggle here so we can give better error messages
        try:
            import kaggle
        except ImportError:
            raise Exception("Kaggle library not installed. Run: pip install kaggle")

        # Check for Kaggle credentials
        kaggle_config_path = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_config_path.exists():
            raise Exception(f"""
Kaggle credentials not found. Please:
1. Go to https://www.kaggle.com/settings/account
2. Click 'Create New Token' to download kaggle.json
3. Create directory: mkdir -p ~/.kaggle
4. Move file: mv ~/Downloads/kaggle.json ~/.kaggle/
5. Set permissions: chmod 600 ~/.kaggle/kaggle.json
""")

        # Download dataset from Kaggle
        dataset_name = "harsh03/sexually-explicit-comments"
        logger.info(f"Downloading {dataset_name} from Kaggle...")

        # Create temporary download directory
        temp_dir = output_path / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Download using Kaggle API
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(temp_dir),
            unzip=True
        )

        logger.info("✅ Successfully downloaded from Kaggle")

        # Find the CSV file(s) in the downloaded data
        csv_files = list(temp_dir.glob("*.csv"))
        if not csv_files:
            raise Exception("No CSV files found in downloaded dataset")

        # Use the first CSV file found
        csv_file = csv_files[0]
        logger.info(f"Processing file: {csv_file.name}")

        # Load and process data with flexible parsing
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.ParserError:
            # Try reading with different parameters if parsing fails
            logger.warning("Standard CSV parsing failed, trying alternative methods...")
            try:
                # Try reading as single column (no headers)
                df = pd.read_csv(csv_file, header=None, names=['text'])
            except:
                # Last resort: read as text lines
                with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                df = pd.DataFrame({'text': lines})

        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Dataset columns: {list(df.columns)}")

        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampling {sample_size} examples for testing")

        # Find text column
        text_column = None
        for col in ['text', 'comment', 'content', 'message', 'tweet']:
            if col in df.columns:
                text_column = col
                break

        if text_column is None:
            # If no obvious text column, use the first non-numeric column
            for col in df.columns:
                if df[col].dtype == 'object':  # String column
                    text_column = col
                    break

        if text_column is None:
            raise ValueError(f"No text column found. Available columns: {list(df.columns)}")

        logger.info(f"Using text column: {text_column}")

        # Convert dataset labels to our 8-label schema
        logger.info("Converting dataset labels to 8-label schema...")
        processed_data = []

        for idx, row in df.iterrows():
            # Convert dataset labels to our 8-label schema
            labels_dict = map_sexual_labels(row)

            # Create label vector
            label_vector = create_label_vector(labels_dict)

            # Create processed row
            processed_row = {
                'text': str(row[text_column]),
                'source': 'kaggle_sexual_content',
                'original_id': row.get('id', f'sexual_{idx}'),
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

        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)

        # Create summary statistics
        create_dataset_summary(processed_df, output_path)

        logger.info(f"✅ Sexual content dataset successfully downloaded and processed!")
        logger.info(f"📊 Total samples: {len(processed_df)}")
        logger.info(f"📁 Output directory: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"❌ Failed to download sexual content dataset: {str(e)}")
        raise


def create_dataset_summary(df: pd.DataFrame, output_path: Path):
    """Create summary statistics for the dataset."""
    summary_file = output_path / "dataset_summary.txt"

    # Calculate label statistics
    label_columns = ['toxicity', 'hate_speech', 'harassment', 'self_harm',
                    'violence', 'sexual', 'profanity', 'spam']

    with open(summary_file, 'w') as f:
        f.write(f"# Sexual Content Dataset Summary\n\n")
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
    parser = argparse.ArgumentParser(description="Download sexual content dataset from Kaggle")
    parser.add_argument("--sample-size", type=int, help="Limit number of samples for testing")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for datasets")

    args = parser.parse_args()

    download_sexual_content(args.output_dir, args.sample_size)

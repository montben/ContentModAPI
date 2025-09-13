"""
Download TweetEval datasets.

This script downloads TweetEval datasets from HuggingFace Hub and processes them
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


def map_tweeteval_labels(row: Dict, task: str) -> Dict[str, bool]:
    """
    Map TweetEval dataset labels to our 8-label schema.

    TweetEval has different tasks: hate, offensive, etc.
    Each has binary labels (0/1)
    """
    label = row.get('label', 0)
    is_positive = bool(label)  # 1 = hate/offensive, 0 = not

    if task == 'hate':
        return {
            "toxicity": is_positive,
            "hate_speech": is_positive,
            "harassment": is_positive,  # Hate speech often involves harassment
            "self_harm": False,
            "violence": False,
            "sexual": False,
            "profanity": False,
            "spam": False
        }
    elif task == 'offensive':
        return {
            "toxicity": is_positive,
            "hate_speech": False,  # Offensive ≠ hate speech
            "harassment": is_positive,  # Offensive language can be harassment
            "self_harm": False,
            "violence": is_positive,  # Some offensive content is violent
            "sexual": False,  # Would need text analysis
            "profanity": is_positive,  # Offensive language often includes profanity
            "spam": False
        }
    else:
        # Default mapping for unknown tasks
        return {
            "toxicity": is_positive,
            "hate_speech": False,
            "harassment": False,
            "self_harm": False,
            "violence": False,
            "sexual": False,
            "profanity": False,
            "spam": False
        }


def download_tweeteval(output_dir: str, sample_size: Optional[int] = None) -> Path:
    """
    Download and process TweetEval datasets.

    Args:
        output_dir: Directory to save the dataset
        sample_size: Optional limit on number of samples (for testing)

    Returns:
        Path to the downloaded dataset directory
    """
    output_path = Path(output_dir) / "tweeteval"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading TweetEval datasets to {output_path}")

    # TweetEval tasks to download
    tasks = ['hate', 'offensive']
    all_processed_data = []

    for task in tasks:
        try:
            logger.info(f"Loading TweetEval {task} dataset...")
            dataset = load_dataset('tweet_eval', task)
            logger.info(f"✅ Successfully loaded tweet_eval {task}")

            # Process training data
            train_data = dataset['train']
            logger.info(f"Loaded {len(train_data)} samples from {task} task")

            # Convert to pandas DataFrame
            df = train_data.to_pandas()

            # Sample if requested (per task)
            task_sample_size = sample_size // len(tasks) if sample_size else None
            if task_sample_size and task_sample_size < len(df):
                df = df.sample(n=task_sample_size, random_state=42).reset_index(drop=True)
                logger.info(f"Sampling {task_sample_size} examples from {task} task")

            # Process each row
            for idx, row in df.iterrows():
                # Convert dataset labels to our 8-label schema
                labels_dict = map_tweeteval_labels(row, task)

                # Create label vector
                label_vector = create_label_vector(labels_dict)

                # Create processed row
                processed_row = {
                    'text': row['text'],
                    'source': f'tweeteval_{task}',
                    'original_id': f'{task}_{idx}',
                    # Add individual label columns
                    **labels_dict,
                    # Add label vector as string for CSV storage
                    'label_vector': ','.join(map(str, label_vector))
                }

                all_processed_data.append(processed_row)

            # Progress logging
            logger.info(f"Processed {len(df)} samples from {task} task")

        except Exception as e:
            logger.error(f"❌ Failed to load {task} task: {e}")
            continue

    if not all_processed_data:
        logger.error("❌ No TweetEval datasets could be loaded")
        raise Exception("No available TweetEval datasets found")

    # Convert to DataFrame
    processed_df = pd.DataFrame(all_processed_data)

    # Save processed data
    processed_file = output_path / "processed_data.csv"
    processed_df.to_csv(processed_file, index=False)
    logger.info(f"Saved processed data to {processed_file}")

    # Create summary statistics
    create_dataset_summary(processed_df, output_path)

    logger.info(f"✅ TweetEval datasets successfully downloaded and processed!")
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
        f.write(f"# TweetEval Dataset Summary\n\n")
        f.write(f"Total samples: {len(df):,}\n\n")

        f.write("## Label Distribution:\n")
        for label in label_columns:
            count = df[label].sum()
            percentage = (count / len(df)) * 100
            f.write(f"- {label}: {count:,} ({percentage:.2f}%)\n")

        f.write("\n## Source Distribution:\n")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"- {source}: {count:,} ({percentage:.2f}%)\n")

        f.write("\n## Text Statistics:\n")
        text_lengths = df['text'].str.len()
        f.write(f"- Average length: {text_lengths.mean():.2f} characters\n")
        f.write(f"- Median length: {text_lengths.median():.0f} characters\n")
        f.write(f"- Min length: {text_lengths.min()} characters\n")
        f.write(f"- Max length: {text_lengths.max()} characters\n")

    logger.info(f"📈 Dataset summary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TweetEval datasets")
    parser.add_argument("--sample-size", type=int, help="Limit number of samples for testing")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for datasets")

    args = parser.parse_args()

    download_tweeteval(args.output_dir, args.sample_size)

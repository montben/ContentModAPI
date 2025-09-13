"""
Download additional TweetEval datasets for missing labels.

This script downloads TweetEval emotion and stance_feminist tasks from HuggingFace Hub
and processes them for use in the Muzzle content moderation pipeline.
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


def map_emotion_labels(row: Dict) -> Dict[str, bool]:
    """
    Map TweetEval emotion labels to our 8-label schema.

    Emotion labels: ['anger', 'joy', 'optimism', 'sadness']
    Sadness might indicate self-harm risk.
    """
    label_names = ['anger', 'joy', 'optimism', 'sadness']
    label = row.get('label', 0)
    emotion = label_names[label] if label < len(label_names) else 'unknown'

    is_sadness = (emotion == 'sadness')
    is_anger = (emotion == 'anger')

    return {
        "toxicity": is_anger,  # Anger can be toxic
        "hate_speech": False,
        "harassment": is_anger,  # Angry content might be harassment
        "self_harm": is_sadness,  # Sadness might indicate self-harm risk
        "violence": is_anger,  # Angry content might be violent
        "sexual": False,
        "profanity": is_anger,  # Angry content often includes profanity
        "spam": False
    }


def map_feminist_stance_labels(row: Dict) -> Dict[str, bool]:
    """
    Map TweetEval stance_feminist labels to our 8-label schema.

    Stance labels: ['none', 'against', 'favor']
    'against' feminism might indicate misogyny/sexual harassment.
    """
    label_names = ['none', 'against', 'favor']
    label = row.get('label', 0)
    stance = label_names[label] if label < len(label_names) else 'unknown'

    is_against_feminism = (stance == 'against')

    return {
        "toxicity": is_against_feminism,  # Anti-feminist content can be toxic
        "hate_speech": is_against_feminism,  # Anti-feminist hate
        "harassment": is_against_feminism,  # Often involves harassment
        "self_harm": False,
        "violence": False,
        "sexual": is_against_feminism,  # Anti-feminist content often includes sexual harassment
        "profanity": False,
        "spam": False
    }


def download_additional_tweeteval(output_dir: str, sample_size: Optional[int] = None) -> Path:
    """
    Download and process additional TweetEval datasets for missing labels.

    Args:
        output_dir: Directory to save the dataset
        sample_size: Optional limit on number of samples (for testing)

    Returns:
        Path to the downloaded dataset directory
    """
    output_path = Path(output_dir) / "tweeteval_additional"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading additional TweetEval datasets to {output_path}")

    # TweetEval tasks for missing labels
    tasks_config = [
        ('emotion', map_emotion_labels, 'For self-harm detection via sadness'),
        ('stance_feminist', map_feminist_stance_labels, 'For sexual harassment detection via anti-feminist stance')
    ]

    all_processed_data = []

    for task, mapper_func, description in tasks_config:
        try:
            logger.info(f"Loading TweetEval {task} dataset - {description}...")
            dataset = load_dataset('tweet_eval', task)
            logger.info(f"✅ Successfully loaded tweet_eval {task}")

            # Process training data
            train_data = dataset['train']
            logger.info(f"Loaded {len(train_data)} samples from {task} task")

            # Convert to pandas DataFrame
            df = train_data.to_pandas()

            # Sample if requested (per task)
            task_sample_size = sample_size // len(tasks_config) if sample_size else None
            if task_sample_size and task_sample_size < len(df):
                df = df.sample(n=task_sample_size, random_state=42).reset_index(drop=True)
                logger.info(f"Sampling {task_sample_size} examples from {task} task")

            # Process each row
            for idx, row in df.iterrows():
                # Convert dataset labels to our 8-label schema
                labels_dict = mapper_func(row)

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
        logger.error("❌ No additional TweetEval datasets could be loaded")
        raise Exception("No available additional TweetEval datasets found")

    # Convert to DataFrame
    processed_df = pd.DataFrame(all_processed_data)

    # Save processed data
    processed_file = output_path / "processed_data.csv"
    processed_df.to_csv(processed_file, index=False)
    logger.info(f"Saved processed data to {processed_file}")

    # Create summary statistics
    create_dataset_summary(processed_df, output_path)

    logger.info(f"✅ Additional TweetEval datasets successfully downloaded and processed!")
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
        f.write(f"# Additional TweetEval Dataset Summary\n\n")
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
    parser = argparse.ArgumentParser(description="Download additional TweetEval datasets for missing labels")
    parser.add_argument("--sample-size", type=int, help="Limit number of samples for testing")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for datasets")

    args = parser.parse_args()

    download_additional_tweeteval(args.output_dir, args.sample_size)

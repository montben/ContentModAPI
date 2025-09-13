"""
Download hate speech/toxicity dataset for content moderation.

This script downloads the hate_speech_offensive dataset from HuggingFace Hub
and processes it for use in the Muzzle content moderation pipeline.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocessing.label_schema import map_jigsaw_labels, map_hate_speech_offensive_labels, create_label_vector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_jigsaw(output_dir: str, sample_size: Optional[int] = None) -> Path:
    """
    Download and process hate speech/toxicity dataset.

    Args:
        output_dir: Directory to save the dataset
        sample_size: Optional limit on number of samples (for testing)

    Returns:
        Path to the downloaded dataset directory
    """
    output_path = Path(output_dir) / "jigsaw"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading hate speech dataset to {output_path}")

    try:
        # Load hate speech dataset (the only one that works)
        logger.info("Loading hate_speech_offensive dataset from HuggingFace Hub...")

        try:
            dataset = load_dataset("hate_speech_offensive")
            logger.info("✅ Successfully loaded hate_speech_offensive")
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            logger.warning("Creating sample data for testing...")
            return create_sample_data(output_path, sample_size or 1000)

        # Process training data
        train_data = dataset['train']
        logger.info(f"Loaded {len(train_data)} training samples")

        # Convert to pandas DataFrame for easier processing
        df = train_data.to_pandas()

        # Sample data if requested (useful for testing)
        if sample_size and sample_size < len(df):
            logger.info(f"Sampling {sample_size} examples for testing")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # Process the data
        processed_data = []

        logger.info("Converting dataset labels to 8-label schema...")
        logger.info(f"Dataset columns: {list(df.columns)}")

        # Determine text column name
        text_column = None
        for col in ['comment_text', 'text', 'tweet', 'comment']:
            if col in df.columns:
                text_column = col
                break

        if text_column is None:
            raise ValueError(f"No text column found. Available columns: {list(df.columns)}")

        logger.info(f"Using text column: {text_column}")

        for idx, row in df.iterrows():
            # Convert dataset labels to our 8-label schema
            labels_dict = map_hate_speech_offensive_labels(row)

            # Create label vector
            label_vector = create_label_vector(labels_dict)

            # Create processed row
            processed_row = {
                'text': row[text_column],
                'source': 'hate_speech_offensive',
                'original_id': row.get('id', f'sample_{idx}'),
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
        output_file = output_path / "processed_data.csv"
        processed_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")

        # Save raw data as well (for reference)
        raw_file = output_path / "raw_data.csv"
        df.to_csv(raw_file, index=False)
        logger.info(f"Saved raw data to {raw_file}")

        # Create summary statistics
        create_dataset_summary(processed_df, output_path)

        logger.info(f"✅ Hate speech dataset successfully downloaded and processed!")
        logger.info(f"📊 Total samples: {len(processed_df)}")
        logger.info(f"📁 Output directory: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"❌ Failed to download Jigsaw dataset: {str(e)}")
        raise


def create_sample_data(output_path: Path, num_samples: int) -> Path:
    """Create sample toxicity data for testing when real datasets aren't available."""

    logger.info(f"Creating {num_samples} sample toxic/safe comments for testing...")

    # Sample toxic and safe comments for testing
    toxic_samples = [
        "You are such an idiot and should go kill yourself",
        "I hate all people like you, you're disgusting",
        "This is complete garbage and you're a moron",
        "Go die in a fire you worthless piece of trash",
        "I want to hurt you badly for saying that",
        "You're so stupid it makes me sick",
        "All [group] people are terrible and should be banned",
        "I hope something bad happens to you",
        "You deserve to suffer for your opinion",
        "This makes me want to punch someone"
    ]

    safe_samples = [
        "I disagree with your opinion but respect your right to have it",
        "This is a really interesting article about technology",
        "I love spending time with my family on weekends",
        "The weather has been quite nice lately",
        "This movie was entertaining and well-made",
        "I enjoy reading books about history and science",
        "Thank you for sharing this helpful information",
        "I appreciate your thoughtful response to my question",
        "This recipe looks delicious, I'll try making it",
        "I'm looking forward to the weekend plans"
    ]

    # Create balanced dataset
    sample_data = []
    samples_per_type = num_samples // 2

    # Add toxic samples
    for i in range(samples_per_type):
        text = toxic_samples[i % len(toxic_samples)]
        sample_data.append({
            'text': text,
            'source': 'sample',
            'original_id': f'toxic_{i}',
            'toxicity': True,
            'hate_speech': i % 3 == 0,  # Some toxic content is hate speech
            'harassment': i % 4 == 0,   # Some is harassment
            'self_harm': i % 10 == 0,   # Rare self-harm content
            'violence': i % 5 == 0,     # Some violent content
            'sexual': False,            # No sexual content in samples
            'profanity': i % 2 == 0,    # Half have profanity
            'spam': False,              # No spam in samples
            'label_vector': None        # Will be calculated
        })

    # Add safe samples
    for i in range(samples_per_type):
        text = safe_samples[i % len(safe_samples)]
        sample_data.append({
            'text': text,
            'source': 'sample',
            'original_id': f'safe_{i}',
            'toxicity': False,
            'hate_speech': False,
            'harassment': False,
            'self_harm': False,
            'violence': False,
            'sexual': False,
            'profanity': False,
            'spam': False,
            'label_vector': None
        })

    # Calculate label vectors
    for item in sample_data:
        labels_dict = {k: v for k, v in item.items()
                      if k in ['toxicity', 'hate_speech', 'harassment', 'self_harm',
                              'violence', 'sexual', 'profanity', 'spam']}
        label_vector = create_label_vector(labels_dict)
        item['label_vector'] = ','.join(map(str, label_vector))

    # Create DataFrame and save
    df = pd.DataFrame(sample_data)

    # Save processed data
    output_file = output_path / "processed_data.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved sample data to {output_file}")

    # Create summary
    create_dataset_summary(df, output_path)

    logger.info(f"✅ Sample dataset created successfully!")
    logger.info(f"📊 Total samples: {len(df)} (for testing purposes)")

    return output_path


def create_dataset_summary(df: pd.DataFrame, output_path: Path):
    """Create a summary of the dataset statistics."""

    # Label columns (our 8-label schema)
    label_columns = ['toxicity', 'hate_speech', 'harassment', 'self_harm',
                    'violence', 'sexual', 'profanity', 'spam']

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

    # Save summary
    summary_file = output_path / "dataset_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("# Jigsaw Dataset Summary\n\n")
        f.write(f"Total samples: {stats['total_samples']:,}\n\n")
        f.write("## Label Distribution:\n")
        for label, info in stats['label_distribution'].items():
            f.write(f"- {label}: {info['count']:,} ({info['percentage']}%)\n")
        f.write(f"\n## Text Statistics:\n")
        f.write(f"- Average length: {stats['text_statistics']['avg_length']} characters\n")
        f.write(f"- Median length: {stats['text_statistics']['median_length']} characters\n")
        f.write(f"- Min length: {stats['text_statistics']['min_length']} characters\n")
        f.write(f"- Max length: {stats['text_statistics']['max_length']} characters\n")

    logger.info(f"📈 Dataset summary saved to {summary_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Jigsaw dataset")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for dataset")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size for testing (optional)")

    args = parser.parse_args()

    try:
        download_jigsaw(args.output_dir, args.sample_size)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

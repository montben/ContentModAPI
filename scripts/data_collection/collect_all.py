"""
Master script to download and collect all public datasets for Muzzle training.

This script downloads 6 public datasets:
1. Jigsaw Toxic Comment Classification
2. HatEval (SemEval-2019 Task 5)
3. Davidson et al. Hate Speech Dataset
4. Founta et al. (2018) - Abusive Language Detection
5. OffensEval (SemEval-2019 Task 6)
6. HASOC (Hate Speech and Offensive Content Identification)

Usage:
    python scripts/data_collection/collect_all.py
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_collection.download_jigsaw import download_jigsaw
from scripts.data_collection.download_hateval import download_hateval
from scripts.data_collection.download_davidson import download_davidson
# TODO: Import other dataset downloaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configuration
DATASETS = {
    "jigsaw": {
        "name": "Jigsaw Toxic Comment Classification",
        "download_func": download_jigsaw,
        "expected_size": "~160K samples",
        "labels": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    },
    "hateval": {
        "name": "HatEval (SemEval-2019 Task 5)",
        "download_func": download_hateval,
        "expected_size": "~20K samples",
        "labels": ["hate", "target_classification", "aggressive"]
    },
    "davidson": {
        "name": "Davidson et al. Hate Speech Dataset",
        "download_func": download_davidson,
        "expected_size": "~25K samples",
        "labels": ["hate_speech", "offensive_language", "neither"]
    }
    # TODO: Add other datasets
}


def main():
    """Main function to orchestrate dataset collection."""
    parser = argparse.ArgumentParser(description="Download all public datasets for Muzzle training")
    parser.add_argument("--data-dir", type=str, default="data/datasets",
                       help="Directory to store downloaded datasets")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                       default=list(DATASETS.keys()),
                       help="Specific datasets to download")
    parser.add_argument("--force", action="store_true",
                       help="Re-download even if dataset already exists")

    args = parser.parse_args()

    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting dataset collection to {data_dir}")
    logger.info(f"Downloading {len(args.datasets)} datasets: {', '.join(args.datasets)}")

    results = {}

    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            continue

        dataset_info = DATASETS[dataset_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading: {dataset_info['name']}")
        logger.info(f"Expected size: {dataset_info['expected_size']}")
        logger.info(f"Labels: {', '.join(dataset_info['labels'])}")
        logger.info(f"{'='*60}")

        try:
            # Check if dataset already exists
            dataset_path = data_dir / dataset_name
            if dataset_path.exists() and not args.force:
                logger.info(f"Dataset {dataset_name} already exists. Use --force to re-download.")
                results[dataset_name] = {"status": "skipped", "path": str(dataset_path)}
                continue

            # Download the dataset
            download_func = dataset_info["download_func"]
            result_path = download_func(output_dir=str(data_dir))

            results[dataset_name] = {
                "status": "success",
                "path": str(result_path),
                "info": dataset_info
            }
            logger.info(f"✅ Successfully downloaded {dataset_name}")

        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_name}: {str(e)}")
            results[dataset_name] = {"status": "failed", "error": str(e)}

    # Summary report
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")

    successful = [k for k, v in results.items() if v["status"] == "success"]
    skipped = [k for k, v in results.items() if v["status"] == "skipped"]
    failed = [k for k, v in results.items() if v["status"] == "failed"]

    logger.info(f"✅ Successful: {len(successful)} - {', '.join(successful)}")
    logger.info(f"⏭️  Skipped: {len(skipped)} - {', '.join(skipped)}")
    logger.info(f"❌ Failed: {len(failed)} - {', '.join(failed)}")

    if successful or skipped:
        logger.info(f"\n📁 Downloaded datasets are in: {data_dir}")
        logger.info("Next step: Run preprocessing to merge datasets")
        logger.info("Command: python scripts/preprocessing/merge_datasets.py")

    return len(failed) == 0  # Return True if no failures


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
Simple data collection script for Muzzle content moderation.

This script downloads the Civil Comments dataset - a comprehensive,
single dataset that covers all core content moderation labels.

No need for multiple datasets anymore!
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_collection.download_civil_comments import download_civil_comments


def main():
    """Download Civil Comments dataset."""
    print("=" * 60)
    print("Muzzle Content Moderation - Data Collection")
    print("=" * 60)
    print()
    print("Downloading Civil Comments dataset...")
    print("This is a comprehensive dataset with 300k+ samples")
    print("covering all 6 core content moderation labels.")
    print()

    # Download full dataset
    # Use sample_size parameter for testing (e.g., 10000)
    output_dir = "data/datasets"

    try:
        result_path = download_civil_comments(
            output_dir=output_dir,
            sample_size=None  # None = download all data
        )

        print()
        print("=" * 60)
        print("✅ Data collection complete!")
        print(f"📁 Dataset saved to: {result_path}")
        print("=" * 60)
        print()
        print("Next step: Run training with scripts/training/train_bert.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()


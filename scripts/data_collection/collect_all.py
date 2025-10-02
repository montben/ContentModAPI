"""
DEPRECATED: This script collected multiple datasets.

The project has been simplified to use a single comprehensive dataset.
Use scripts/data_collection/collect_data.py instead.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_collection.collect_data import main as collect_main


def main():
    print("⚠️  DEPRECATED: collect_all.py is deprecated.")
    print("📝 The project now uses a single comprehensive dataset.")
    print("🔄 Redirecting to collect_data.py...\n")

    return collect_main()


if __name__ == "__main__":
    main()

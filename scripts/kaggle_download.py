from __future__ import annotations

import logging
import shutil
from pathlib import Path

import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Project Paths
# ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DATA_DIR / "Assessment_TwitterDataset.csv"

KAGGLE_DATASET = "arindamsahoo/social-media-users"


def download_dataset() -> Path:

    if OUTPUT_FILE.exists():
        logger.info("Dataset already exists: %s", OUTPUT_FILE)
        return OUTPUT_FILE

    logger.info("Downloading dataset from Kaggle...")

    dataset_path = Path(
        kagglehub.dataset_download(KAGGLE_DATASET)
    )

    logger.info("Downloaded to cache: %s", dataset_path)

    csv_files = list(dataset_path.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found inside downloaded dataset."
        )

    if len(csv_files) > 1:
        logger.warning(
            "Multiple CSV files found. Using %s",
            csv_files[0].name,
        )

    source = csv_files[0]

    shutil.copy2(source, OUTPUT_FILE)

    logger.info("Dataset copied to %s", OUTPUT_FILE)

    return OUTPUT_FILE


def main() -> None:
    path = download_dataset()

    logger.info("Ready for ingestion: %s", path)


if __name__ == "__main__":
    main()
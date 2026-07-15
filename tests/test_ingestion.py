from pathlib import Path

from src.retrieval.utils.logger import logger
from src.ingestion.loader import load_users
from src.ingestion.splitter import split_dataset

DATA_PATH = Path("data/raw/Assessment_TwitterDataset.csv")


def main():
    logger.info("========== Testing Ingestion Pipeline ==========")

    users = load_users(DATA_PATH)

    train, val, test = split_dataset(users)

    logger.info("First user: %s", train[0])

    logger.info("========== Ingestion Test Completed ==========")


if __name__ == "__main__":
    main()
from pathlib import Path

from src.retrieval.utils.logger import logger
from src.ingestion.loader import load_users
from src.preprocessing.pipeline import preprocess_users
from src.ranking.train import train

DATA_PATH = Path("data/processed/geouser.csv")


def main() -> None:

    logger.info("========== Testing Preprocessing Pipeline ==========")

    users = load_users(DATA_PATH)


    processed_users = preprocess_users(users)
    
    logger.info("=" * 60)
    logger.info(f"Total Users      : {len(users)}")
    logger.info(f"Processed Users  : {len(processed_users)}")
    logger.info("=" * 60)

    logger.info("\nFirst Processed User\n")
    logger.info("First processed user: %s", processed_users[0])
    logger.info("========== Preprocessing Test Completed ==========")

if __name__ == "__main__":
    main()
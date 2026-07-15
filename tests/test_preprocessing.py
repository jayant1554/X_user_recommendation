from pathlib import Path

from src.ingestion.loader import load_users
from src.preprocessing.pipeline import preprocess_users

DATA_PATH = Path("data/raw/Assessment_TwitterDataset.csv")


def main() -> None:

    users = load_users(DATA_PATH)

    processed_users = preprocess_users(users)

    print("=" * 60)
    print(f"Total Users      : {len(users)}")
    print(f"Processed Users  : {len(processed_users)}")
    print("=" * 60)

    print("\nFirst Processed User\n")
    print(processed_users[0])


if __name__ == "__main__":
    main()
from torch.utils.data import DataLoader

from src.ingestion.loader import load_users
from src.preprocessing.pipeline import preprocess_users
from src.ranking.pair_sampler import PairSampler
from src.ranking.encoder import FeatureEncoder
from src.ranking.dataset import (
    RankingDataset,
    ranking_collate_fn,
)


def main():

    users = load_users(
        "data/raw/Assessment_TwitterDataset.csv"
    )

    users = preprocess_users(users)

    encoder = FeatureEncoder()

    encoder.fit(users)

    sampler = PairSampler(users, negatives_per_positive=2, seed=42)

    dataset = RankingDataset(
        sampler,
        encoder,
    )

    print(f"Pairs : {len(dataset)}")

    sample = dataset[0]

    print(sample["target"].keys())

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=ranking_collate_fn,
    )

    batch = next(iter(loader))

    print(batch["target"]["interest_ids"].shape)
    print(batch["candidate"]["interest_ids"].shape)
    print(batch["label"].shape)


if __name__ == "__main__":
    main()
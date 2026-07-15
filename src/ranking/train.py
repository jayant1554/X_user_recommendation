
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.ingestion.loader import load_users
from src.preprocessing.pipeline import preprocess_users
from src.ingestion.splitter import split_dataset
from src.ranking.dataset import RankingDataset, ranking_collate_fn
from src.ranking.encoder import FeatureEncoder
from src.ranking.evaluate import (
    _collect_scores_and_labels,
    evaluate_classification_metrics,
    find_best_threshold,
)
from src.ranking.model import TwoTowerModel
from src.ranking.pair_sampler import PairSampler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/Assessment_TwitterDataset.csv"
MODEL_DIR = Path("models")
MAX_INTERESTS = 10
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 1e-3
NEGATIVES_PER_POSITIVE = 2


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    raw_users = load_users(DATA_PATH)
    users = preprocess_users(raw_users)
    train_users, val_users, test_users = split_dataset(users)

    # Encoder is fit on train only, to avoid val/test leakage into the
    # learned vocab/scaler state.
    encoder = FeatureEncoder(max_interests=MAX_INTERESTS)
    encoder.fit(train_users)

    train_sampler = PairSampler(train_users, negatives_per_positive=NEGATIVES_PER_POSITIVE, seed=42)
    val_sampler = PairSampler(val_users, negatives_per_positive=NEGATIVES_PER_POSITIVE, seed=123)
    test_sampler = PairSampler(test_users, negatives_per_positive=NEGATIVES_PER_POSITIVE, seed=456)

    train_ds = RankingDataset(train_sampler, encoder)
    val_ds = RankingDataset(val_sampler, encoder)
    test_ds = RankingDataset(test_sampler, encoder)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ranking_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ranking_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ranking_collate_fn)

    model = TwoTowerModel(
        num_interests=encoder.num_interests,
        num_genders=encoder.num_genders,
        num_countries=encoder.num_countries,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            target = {k: v.to(device) for k, v in batch["target"].items()}
            candidate = {k: v.to(device) for k, v in batch["candidate"].items()}
            label = batch["label"].to(device)

            optimizer.zero_grad()
            preds = model(target, candidate)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(label)

        avg_train_loss = total_loss / len(train_ds)
        val_metrics = evaluate_classification_metrics(model, val_loader, device)
        logger.info("Epoch %d/%d - train_loss=%.4f - val: %s", epoch, EPOCHS, avg_train_loss, val_metrics)

   
    val_scores, val_labels = _collect_scores_and_labels(model, val_loader, device)
    best_threshold, best_val_f1 = find_best_threshold(val_scores, val_labels)
    logger.info("Calibrated threshold (on val): %.2f -> val F1 at that threshold=%.4f", best_threshold, best_val_f1)

    test_metrics_default = evaluate_classification_metrics(model, test_loader, device, threshold=0.5)
    logger.info("Final test set (threshold=0.50, uncalibrated): %s", test_metrics_default)

    test_metrics_calibrated = evaluate_classification_metrics(model, test_loader, device, threshold=best_threshold)
    logger.info("Final test set (threshold=%.2f, calibrated): %s", best_threshold, test_metrics_calibrated)

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "two_tower.pt")
    encoder.save(MODEL_DIR)
    with open(MODEL_DIR / "model_config.json", "w") as f:
        json.dump({
            "num_interests": encoder.num_interests,
            "num_genders": encoder.num_genders,
            "num_countries": encoder.num_countries,
            "max_interests": MAX_INTERESTS,
            "calibrated_threshold": best_threshold,
        }, f, indent=2)

    logger.info("Saved model + encoder to %s", MODEL_DIR)


if __name__ == "__main__":
    train()
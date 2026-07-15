from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.ingestion.schema import ProcessedUser

logger = logging.getLogger(__name__)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

SKLEARN_UNKNOWN_SENTINEL = -1


class FeatureEncoder:
    """Fit once on the training split, then encode_user() everywhere else
    (train/val/test/live inference) using the same fitted transformers."""

    def __init__(self, max_interests: int = 10) -> None:
        self.max_interests = max_interests
        self.interest_to_idx: dict[str, int] | None = None

        self.gender_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=SKLEARN_UNKNOWN_SENTINEL
        )
        self.country_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=SKLEARN_UNKNOWN_SENTINEL
        )
        self.age_scaler = StandardScaler()
        self._is_fitted = False

    @staticmethod
    def _build_interest_vocab(values: set[str]) -> dict[str, int]:
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for value in sorted(values):
            vocab[value] = len(vocab)
        return vocab

    def fit(self, users: list[ProcessedUser]) -> None:
        interests: set[str] = set()
        genders: list[list[str]] = []
        countries: list[list[str]] = []
        ages: list[list[float]] = []

        for user in users:
            interests.update(user.interests)
            genders.append([user.gender])
            countries.append([user.country])
            ages.append([float(user.age)])

        self.interest_to_idx = self._build_interest_vocab(interests)
        self.gender_encoder.fit(genders)
        self.country_encoder.fit(countries)
        self.age_scaler.fit(ages)

        self._is_fitted = True
        logger.info(
            "Encoder fitted: %d interests, %d genders, %d countries",
            len(self.interest_to_idx),
            len(self.gender_encoder.categories_[0]),
            len(self.country_encoder.categories_[0]),
        )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("FeatureEncoder not fitted -- call fit() first.")

    def _encode_interests(self, interests: list[str]) -> tuple[list[int], list[int]]:
        ids = [self.interest_to_idx.get(i, self.interest_to_idx[UNK_TOKEN]) for i in interests]
        ids = ids[: self.max_interests]
        mask = [1] * len(ids)
        pad = self.max_interests - len(ids)
        ids += [0] * pad
        mask += [0] * pad
        return ids, mask

    def _ordinal_transform(self, encoder: OrdinalEncoder, value: str) -> int:
        code = int(encoder.transform([[value]])[0][0])
        if code == SKLEARN_UNKNOWN_SENTINEL:
            return len(encoder.categories_[0])
        return code

    def encode_user(self, user: ProcessedUser) -> dict[str, torch.Tensor]:
        self._check_fitted()
        interest_ids, interest_mask = self._encode_interests(user.interests)
        gender_id = self._ordinal_transform(self.gender_encoder, user.gender)
        country_id = self._ordinal_transform(self.country_encoder, user.country)
        age_z = float(self.age_scaler.transform([[float(user.age)]])[0][0])

        return {
            "age": torch.tensor(age_z, dtype=torch.float32),
            "gender_id": torch.tensor(gender_id, dtype=torch.long),
            "country_id": torch.tensor(country_id, dtype=torch.long),
            "interest_ids": torch.tensor(interest_ids, dtype=torch.long),
            "interest_mask": torch.tensor(interest_mask, dtype=torch.float32),
        }

    def encode_batch(self, users: list[ProcessedUser]) -> dict[str, torch.Tensor]:
        encoded = [self.encode_user(u) for u in users]
        return {
            key: torch.stack([e[key] for e in encoded])
            for key in ("age", "gender_id", "country_id", "interest_ids", "interest_mask")
        }

    @property
    def num_genders(self) -> int:
        return len(self.gender_encoder.categories_[0]) + 1

    @property
    def num_countries(self) -> int:
        return len(self.country_encoder.categories_[0]) + 1

    @property
    def num_interests(self) -> int:
        return len(self.interest_to_idx)

    def save(self, model_dir: str | Path) -> None:
        self._check_fitted()
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "interest_vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.interest_to_idx, f, indent=2)

        joblib.dump(self.gender_encoder, model_dir / "gender_encoder.pkl")
        joblib.dump(self.country_encoder, model_dir / "country_encoder.pkl")
        joblib.dump(self.age_scaler, model_dir / "age_scaler.pkl")

        with open(model_dir / "encoder_config.json", "w", encoding="utf-8") as f:
            json.dump({"max_interests": self.max_interests}, f, indent=2)

        logger.info("Encoder saved to %s", model_dir)

    @classmethod
    def load(cls, model_dir: str | Path) -> "FeatureEncoder":
        model_dir = Path(model_dir)
        with open(model_dir / "encoder_config.json", encoding="utf-8") as f:
            config = json.load(f)

        encoder = cls(max_interests=config["max_interests"])
        with open(model_dir / "interest_vocab.json", encoding="utf-8") as f:
            encoder.interest_to_idx = json.load(f)

        encoder.gender_encoder = joblib.load(model_dir / "gender_encoder.pkl")
        encoder.country_encoder = joblib.load(model_dir / "country_encoder.pkl")
        encoder.age_scaler = joblib.load(model_dir / "age_scaler.pkl")
        encoder._is_fitted = True

        logger.info("Encoder loaded from %s", model_dir)
        return encoder
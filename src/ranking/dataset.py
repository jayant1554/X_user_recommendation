from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .encoder import FeatureEncoder
from .pair_sampler import PairSampler

class RankingDataset(Dataset):
    def __init__(self, sampler: PairSampler, encoder: FeatureEncoder) -> None:
        self.sampler = sampler
        self.encoder = encoder
        # One encode_batch() call for the whole split, instead of one
        # encode_user() call per pair per epoch -- see module docstring.
        self._encoded = encoder.encode_batch(sampler.users)

    def __len__(self) -> int:
        return len(self.sampler)

    def _lookup(self, user_index: int) -> dict[str, torch.Tensor]:
        """Pulls one user's already-encoded tensors out of the cache
        built in __init__ -- plain indexing, no sklearn calls."""
        return {key: value[user_index] for key, value in self._encoded.items()}

    def __getitem__(self, index: int) -> dict:
        target_idx, candidate_idx, label = self.sampler[index]
        return {
            "target": self._lookup(target_idx),
            "candidate": self._lookup(candidate_idx),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def ranking_collate_fn(batch: list[dict]) -> dict:
    """Stacks a list of {"target": {...}, "candidate": {...}, "label": ...}
    dicts into batched tensors, keyed the same way, for each side."""
    keys = ("age", "gender_id", "country_id", "interest_ids", "interest_mask")

    def _stack_side(side: str) -> dict[str, torch.Tensor]:
        return {key: torch.stack([item[side][key] for item in batch]) for key in keys}

    return {
        "target": _stack_side("target"),
        "candidate": _stack_side("candidate"),
        "label": torch.stack([item["label"] for item in batch]),
    }
"""
Stage 2 — Neural Ranking: Two-Tower model.

Features used: interests, gender, country, age. City is NOT used here --
it lives in Stage 1 retrieval only. Ranking works off a coarser,
lower-cardinality feature set than retrieval; deliberate stage-scope
difference (city has extremely high cardinality relative to dataset size
-- most cities have only 1-2 users -- so it isn't a useful learned
embedding at this scale).

Both target and candidate users are the same kind of entity, so a single
shared UserTower encodes both sides rather than two independently-weighted
towers -- deliberate design choice.

Affinity score: sigmoid(temperature * cosine_similarity(target_emb,
candidate_emb)) -- cosine + learnable temperature instead of a raw dot
product, since raw dot products are unbounded and caused a "most scores
saturate near 1.0" miscalibration during early development. L2-normalizing
both embeddings bounds similarity to [-1, 1] before the sigmoid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


RETRIEVAL_SCORE_WEIGHT = 0.7
NEURAL_SCORE_WEIGHT = 0.3


class UserTower(nn.Module):
    def __init__(
        self,
        num_interests: int,
        num_genders: int,
        num_countries: int,
        interest_dim: int = 32,
        gender_dim: int = 4,
        country_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.interest_emb = nn.Embedding(num_interests, interest_dim, padding_idx=0)
        self.gender_emb = nn.Embedding(num_genders, gender_dim)
        self.country_emb = nn.Embedding(num_countries, country_dim)

        combined_dim = interest_dim + gender_dim + country_dim + 1  # +1 for age scalar
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self.interest_emb(features["interest_ids"])            # (B, L, D)
        mask = features["interest_mask"].unsqueeze(-1)                # (B, L, 1)
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        interest_vec = summed / counts                                # (B, D)

        gender_vec = self.gender_emb(features["gender_id"])
        country_vec = self.country_emb(features["country_id"])
        age_vec = features["age"].unsqueeze(-1)                       # (B, 1)

        combined = torch.cat([interest_vec, gender_vec, country_vec, age_vec], dim=-1)
        return self.mlp(combined)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_interests: int,
        num_genders: int,
        num_countries: int,
        interest_dim: int = 32,
        gender_dim: int = 4,
        country_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self._tower = UserTower(
            num_interests=num_interests,
            num_genders=num_genders,
            num_countries=num_countries,
            interest_dim=interest_dim,
            gender_dim=gender_dim,
            country_dim=country_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.temperature = nn.Parameter(torch.tensor(4.0))

    @property
    def target_tower(self) -> UserTower:
        return self._tower

    @property
    def candidate_tower(self) -> UserTower:
        return self._tower

    def encode(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._tower(features)

    def forward(
        self,
        target: dict[str, torch.Tensor],
        candidate: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Score (target, candidate) pairs -- one per row (training)."""
        t_emb = F.normalize(self.encode(target), dim=-1)
        c_emb = F.normalize(self.encode(candidate), dim=-1)
        cosine_sim = (t_emb * c_emb).sum(dim=-1)
        return torch.sigmoid(self.temperature * cosine_sim)

    def score_candidates(
        self,
        target: dict[str, torch.Tensor],    
        candidates: dict[str, torch.Tensor],  
    ) -> torch.Tensor:
        """Inference path: one target vs. N candidates, single batched matmul."""
        t_emb = F.normalize(self.encode(target), dim=-1) 
        c_emb = F.normalize(self.encode(candidates), dim=-1)
        cosine_sim = (c_emb @ t_emb.T).squeeze(-1)         
        return torch.sigmoid(self.temperature * cosine_sim)

    def rank_candidates(
        self,
        target: dict[str, torch.Tensor],
        candidates: dict[str, torch.Tensor],
        retrieval_scores: list[float] | torch.Tensor,
    ) -> torch.Tensor:
        """Blend the rule-based retrieval score with the neural score.

        The model file owns the ranking math; the API layer just prepares
        tensors and turns the final scores into a response.
        """

        with torch.no_grad():
            neural_scores = self.score_candidates(target, candidates)

        if not torch.is_tensor(retrieval_scores):
            retrieval_scores = torch.tensor(
                retrieval_scores,
                dtype=neural_scores.dtype,
                device=neural_scores.device,
            )
        else:
            retrieval_scores = retrieval_scores.to(
                device=neural_scores.device,
                dtype=neural_scores.dtype,
            )

        return (
            RETRIEVAL_SCORE_WEIGHT * retrieval_scores
            + NEURAL_SCORE_WEIGHT * neural_scores
        )
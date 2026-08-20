from __future__ import annotations
import random
from collections import defaultdict

from requests.packages import target

from src.ingestion.schema import ProcessedUser
from src.retrieval.utils.logger import logger
from src.retrieval.candidate_gen import haversine_distance

MAX_CANDIDATES_PER_INTEREST = 500
def jaccard_similarity(a: ProcessedUser, b: ProcessedUser) -> float:
    a_set = set(a.interests)
    b_set = set(b.interests)

    if not a_set or not b_set:
        return 0.0

    return len(a_set & b_set) / len(a_set | b_set)
def is_positive(target: ProcessedUser, candidate: ProcessedUser, max_distance_km: float = 500.0) -> bool:
  distance = haversine_distance(
    target.lat,target.lng,candidate.lat,candidate.lng)
  if distance > max_distance_km:
    return False
  
  return jaccard_similarity(target, candidate) >= 0.30
  

class PairSampler:
    def __init__(
        self,
        users: list[ProcessedUser],
        negatives_per_positive: int = 2,
        max_candidates_per_interest: int = MAX_CANDIDATES_PER_INTEREST,
        seed: int = 42,
    ) -> None:
        self.users = users
        self.negatives_per_positive = negatives_per_positive
        self.max_candidates_per_interest = max_candidates_per_interest
        self.rng = random.Random(seed)

        self.interest_index: dict[str, list[int]] = defaultdict(list)
        for idx, user in enumerate(users):
            for interest in user.interests:
                self.interest_index[interest].append(idx)

        self.pairs = self._build_pairs()
        logger.info("PairSampler: %d users -> %d pairs", len(self.users), len(self.pairs))

    def _positive_candidates(self, user_index: int) -> list[int]:
        user = self.users[user_index]
        candidates: set[int] = set()
        for interest in user.interests:
            bucket = self.interest_index[interest]
            if len(bucket) > self.max_candidates_per_interest:
                bucket = self.rng.sample(bucket, self.max_candidates_per_interest)
            candidates.update(bucket)
        candidates.discard(user_index)
        return [idx for idx in candidates if is_positive(user, self.users[idx])]

    def _sample_negative(self, user_index: int) -> int:
        user = self.users[user_index]
        while True:
            idx = self.rng.randrange(len(self.users))
            if idx == user_index or is_positive(user, self.users[idx]):
                continue
            return idx
    def _build_pairs(self) -> list[tuple[int, int, int]]:
        pairs: list[tuple[int, int, int]] = []
        for idx in range(len(self.users)):
            positives = self._positive_candidates(idx)
            if not positives:
                continue

            positive_idx = self.rng.choice(positives)
            pairs.append((idx, positive_idx, 1))

            for _ in range(self.negatives_per_positive):
                negative_idx = self._sample_negative(idx)
                pairs.append((idx, negative_idx, 0))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[int, int, int]:
        return self.pairs[index]

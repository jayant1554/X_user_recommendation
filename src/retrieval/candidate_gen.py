from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

from src.ingestion.schema import ProcessedUser

INTEREST_WEIGHT = 0.7
LOCATION_WEIGHT = 0.3


@dataclass(slots=True)
class Candidate:
    user: ProcessedUser
    retrieval_score: float


def interest_score(a: ProcessedUser, b: ProcessedUser) -> float:
    inter = len(a.interest_set & b.interest_set)
    if inter == 0:
        return 0.0
    return inter / len(a.interest_set | b.interest_set)


def location_score(a: ProcessedUser, b: ProcessedUser) -> float:
    if a.city == b.city:
        return 1.0
    if a.country == b.country:
        return 0.7
    return 0.0


def combined_score(a: ProcessedUser, b: ProcessedUser) -> float:
    return (
        INTEREST_WEIGHT * interest_score(a, b)
        + LOCATION_WEIGHT * location_score(a, b)
    )


class CandidateGenerator:

    def __init__(self, users: list[ProcessedUser]) -> None:
        self.users = users
        self.user_lookup = {u.user_id: u for u in users}

        self.interest_index: dict[str, list[str]] = defaultdict(list)
        self.city_index: dict[str, list[str]] = defaultdict(list)
        self.country_index: dict[str, list[str]] = defaultdict(list)
        for user in users:
            for interest in user.interests:
                self.interest_index[interest].append(user.user_id)
            self.city_index[user.city].append(user.user_id)
            self.country_index[user.country].append(user.user_id)

    def get_user(self, user_id: str) -> ProcessedUser | None:
        return self.user_lookup.get(user_id)

    def get_candidates(self, target: ProcessedUser, k: int = 100) -> list[Candidate]:

        candidate_ids: set[str] = set()

        for interest in target.interests:
            candidate_ids.update(self.interest_index.get(interest, []))

        candidate_ids.update(self.city_index.get(target.city, []))
        candidate_ids.update(self.country_index.get(target.country, []))

        candidate_ids.discard(target.user_id)

        scored = []

        for user_id in candidate_ids:
            user = self.user_lookup[user_id]
            score = combined_score(target, user)
            scored.append(Candidate(user, score))

        return heapq.nlargest(
            k,
            scored,
            key=lambda x: x.retrieval_score,
        )
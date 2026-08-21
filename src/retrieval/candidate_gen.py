from __future__ import annotations
from math import radians, sin, cos, sqrt, atan2
import heapq
from collections import defaultdict
from dataclasses import dataclass

from requests.packages import target

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

def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    R = 6371.0  # Radius of the Earth in kilometers
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def location_score(a: ProcessedUser, b: ProcessedUser) -> float:
    if (
        a.lat is None
        or a.lng is None
        or b.lat is None
        or b.lng is None
    ):
        return 0.0

    distance = haversine_distance(
        a.lat,
        a.lng,
        b.lat,
        b.lng,
    )

    if distance <= 10:
        return 1.0

    if distance <= 50:
        return 0.9

    if distance <= 100:
        return 0.8

    if distance <= 250:
        return 0.6

    if distance <= 500:
        return 0.4

    if distance <= 1000:
        return 0.2

    return 0.0
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
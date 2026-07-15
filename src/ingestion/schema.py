from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class User:
    user_id: str
    name: str
    gender: str
    dob: str
    interests: list[str]
    city: str
    country: str

    @property
    def interest_set(self) -> set[str]:
        return set(self.interests)


@dataclass(slots=True)
class ProcessedUser:
    user_id: str
    name: str
    gender: str
    age: int
    interests: list[str]
    city: str
    country: str

    @property
    def interest_set(self) -> set[str]:
        return set(self.interests)
from __future__ import annotations

import ast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.ingestion.schema import ProcessedUser
from src.preprocessing.validators import calculate_age, parse_birth_date


def _normalize_text(value: str) -> str:
    return value.strip().lower()


def _normalize_interests(interests: list[str]) -> list[str]:
    cleaned = {_normalize_text(interest) for interest in interests if interest.strip()}
    return sorted(cleaned)


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., example="TEST_USER")
    name: str = Field(..., example="Jayant Bisht")
    gender: str = Field(..., example="Male")
    age: int = Field(..., ge=13)
    interests: list[str] = Field(..., min_length=1)
    city: str = Field(..., example="Delhi")
    country: str = Field(..., example="India")

    def to_processed_user(self) -> ProcessedUser:
        return ProcessedUser(
            user_id=self.user_id,
            name=self.name,
            gender=_normalize_text(self.gender),
            age=self.age,
            interests=_normalize_interests(self.interests),
            city=_normalize_text(self.city),
            country=_normalize_text(self.country),
        )


class RawRecommendationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: str = Field(..., alias="UserID", example="00001")
    name: str = Field(..., alias="Name", example="Jesse Lawhorn")
    gender: str = Field(..., alias="Gender", example="Female")
    dob: str = Field(..., alias="DOB", example="1958-10-15")
    interests: list[str] = Field(..., alias="Interests", min_length=1)
    city: str = Field(..., alias="City", example="Sibolga")
    country: str = Field(..., alias="Country", example="Indonesia")

    @field_validator("interests", mode="before")
    @classmethod
    def _parse_interests(cls, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        if value is None:
            return []

        text = str(value).strip()
        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, str):
                return [parsed.strip()]
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass

        return [
            piece.strip().strip("'").strip('"')
            for piece in text.split(",")
            if piece.strip().strip("'").strip('"')
        ]

    def to_processed_user(self) -> ProcessedUser:
        birth_date = parse_birth_date(self.dob)
        return ProcessedUser(
            user_id=self.user_id,
            name=self.name,
            gender=_normalize_text(self.gender),
            age=calculate_age(birth_date),
            interests=_normalize_interests(self.interests),
            city=_normalize_text(self.city),
            country=_normalize_text(self.country),
        )


class RetrievedCandidate(BaseModel):
    rank: int
    user_id: str
    name: str
    city: str
    country: str
    retrieval_score: float


class RecommendedUser(BaseModel):
    rank: int
    user_id: str
    name: str
    city: str
    country: str
    retrieval_score: float
    final_score: float


class RecommendationResponse(BaseModel):
    retrieved_count: int
    top_k: int
    retrieved: list[RetrievedCandidate]
    recommendations: list[RecommendedUser]


class RecommendationSummaryResponse(BaseModel):
    retrieved_count: int
    top_k: int
    recommendations: list[RecommendedUser]
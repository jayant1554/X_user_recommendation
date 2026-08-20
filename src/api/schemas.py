from __future__ import annotations

import ast
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from src.ingestion.schema import ProcessedUser


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_text(value: str) -> str:
    return value.strip().lower()


def _normalize_interests(interests: list[str]) -> list[str]:
    return sorted({_normalize_text(i) for i in interests if i.strip()})


def _parse_interests_raw(value: object) -> list[str]:
    """Accepts a list, a stringified list/tuple, or a comma-separated string
    and returns a cleaned list of raw (not yet normalized) interest strings.
    Shared by both request schemas so there's exactly one parsing implementation.
    """
    if isinstance(value, list):
        return [str(i).strip() for i in value if str(i).strip()]
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
            return [str(i).strip() for i in parsed if str(i).strip()]
    except (ValueError, SyntaxError):
        pass

    return [s.strip().strip("'\"") for s in text.split(",") if s.strip().strip("'\"")]


def _compute_age(dob: date) -> int:
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    if age < 0:
        raise ValueError("DOB is in the future")
    return age


# ---------------------------------------------------------------------------
# Base config - avoids repeating model_config on every request schema
# ---------------------------------------------------------------------------

class _APIRequestBase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class _UserMixin(_APIRequestBase):
    def to_processed_user(self, lat: float, lng: float) -> ProcessedUser:
        return ProcessedUser(
            user_id=self.user_id,
            name=self.name,
            gender=_normalize_text(self.gender),
            age=self.age,
            interests=_normalize_interests(self.interests),
            city=_normalize_text(self.city),
            country=_normalize_text(self.country),
            lat=round(lat, 6),
            lng=round(lng, 6),
        )


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class RecommendationRequest(_UserMixin):
    user_id: str = Field(..., example="TEST_USER")
    name: str = Field(..., example="Jayant Bisht")
    gender: str = Field(..., example="Male")
    age: int = Field(..., ge=13, le=120)
    interests: list[str] = Field(..., min_length=1, example=["art", "music", "sports"])
    city: str = Field(..., example="Delhi")
    country: str = Field(..., example="India")

    @field_validator("interests", mode="before")
    @classmethod
    def _parse_interests(cls, value: object) -> list[str]:
        return _parse_interests_raw(value)


class RawRecommendationRequest(_UserMixin):
    """Accepts the raw ingestion-style field names (UserID, DOB, etc.) via
    aliases, so no manual property-per-field boilerplate is needed - Pydantic
    handles the name mapping directly."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", populate_by_name=True)

    user_id: str = Field(..., alias="UserID")
    name: str = Field(..., alias="Name")
    gender: str = Field(..., alias="Gender")
    dob: date = Field(..., alias="DOB")
    raw_interests: str = Field(..., alias="Interests")
    city: str = Field(..., alias="City")
    country: str = Field(..., alias="Country")

    @computed_field  # type: ignore[misc]
    @property
    def age(self) -> int:
        return _compute_age(self.dob)

    @computed_field  # type: ignore[misc]
    @property
    def interests(self) -> list[str]:
        return _parse_interests_raw(self.raw_interests)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class RetrievedCandidate(BaseModel):
    rank: int
    user_id: str
    name: str
    city: str
    country: str
    retrieval_score: float


class RecommendedUser(RetrievedCandidate):
    final_score: float


class _RecommendationResultBase(BaseModel):
    retrieved_count: int
    top_k: int
    recommendations: list[RecommendedUser]


class RecommendationResponse(_RecommendationResultBase):
    retrieved: list[RetrievedCandidate]


class RecommendationSummaryResponse(_RecommendationResultBase):
    pass
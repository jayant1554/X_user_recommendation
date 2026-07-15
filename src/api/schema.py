from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., example="TEST_USER")
    name: str = Field(..., example="Jayant Bisht")
    gender: str = Field(..., example="Male")
    age: int = Field(..., ge=13)
    interests: list[str] = Field(..., min_length=1)
    city: str = Field(..., example="Delhi")
    country: str = Field(..., example="India")


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
from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import (
    RecommendationRequest,
    RawRecommendationRequest,
    RecommendationResponse,
    RecommendationSummaryResponse,
)
from src.api.services import rank_recommendations, rank_recommendations_summary

router = APIRouter(tags=["Recommendation"])


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(request: RecommendationRequest):
    return rank_recommendations(request.to_processed_user())


@router.post(
    "/recommend/raw",
    response_model=RecommendationSummaryResponse,
)
def recommend_raw(request: RawRecommendationRequest):
    return rank_recommendations_summary(request.to_processed_user())
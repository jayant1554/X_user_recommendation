from __future__ import annotations

from fastapi import APIRouter, HTTPException
from src.retrieval.utils.logger import logger

from src.api.dependencies import app_state
from src.api.schemas import (
    RawRecommendationRequest,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationSummaryResponse,
)
from src.api.services import (
    rank_recommendations,
    rank_recommendations_summary,
)

router = APIRouter(tags=["Recommendation"])


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(request: RecommendationRequest):

    lat, lng = app_state.geocoder.get_coordinates(
        request.city,
        request.country,
    )
    logger.info(
    "Geocoded %s, %s -> lat=%.6f lng=%.6f",
    request.city,
    request.country,
    lat,
    lng,
)
    user = request.to_processed_user(lat, lng)
    return rank_recommendations(user)


@router.post("/recommend/raw")
def recommend_raw(request: RawRecommendationRequest):

    lat, lng = app_state.geocoder.get_coordinates(
        request.city,
        request.country,
    )
    logger.info(
        "Geocoded %s, %s -> lat=%.6f lng=%.6f",
        request.city,
        request.country,
        lat,
        lng,
    )
    user = request.to_processed_user(lat, lng)
    return rank_recommendations_summary(user)
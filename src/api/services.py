from __future__ import annotations

from src.api.dependencies import app_state
from src.api.schemas import (
    RecommendationResponse,
    RecommendationSummaryResponse,
    RecommendedUser,
    RetrievedCandidate,
)
from src.ingestion.schema import ProcessedUser


def rank_recommendations(
    target: ProcessedUser,
    retrieved_k: int = 100,
    top_k: int = 10,
) -> RecommendationResponse:
    return _rank_recommendations(target, retrieved_k=retrieved_k, top_k=top_k, include_retrieved=True)


def rank_recommendations_summary(
    target: ProcessedUser,
    retrieved_k: int = 100,
    top_k: int = 10,
) -> RecommendationSummaryResponse:
    return _rank_recommendations(target, retrieved_k=retrieved_k, top_k=top_k, include_retrieved=False)


def _rank_recommendations(
    target: ProcessedUser,
    retrieved_k: int,
    top_k: int,
    include_retrieved: bool,
) -> RecommendationResponse | RecommendationSummaryResponse:
    retrieved = app_state.generator.get_candidates(target, k=retrieved_k)

    if not retrieved:
        if include_retrieved:
            return RecommendationResponse(
                retrieved_count=0,
                top_k=0,
                retrieved=[],
                recommendations=[],
            )
        return RecommendationSummaryResponse(
            retrieved_count=0,
            top_k=0,
            recommendations=[],
        )

    target_features = app_state.encoder.encode_user(target)
    target_features = {
        key: value.unsqueeze(0).to(app_state.device)
        for key, value in target_features.items()
    }

    candidate_features = app_state.encoder.encode_batch(
        [candidate.user for candidate in retrieved]
    )
    candidate_features = {
        key: value.to(app_state.device)
        for key, value in candidate_features.items()
    }

    retrieval_scores = [candidate.retrieval_score for candidate in retrieved]
    final_scores = app_state.model.rank_candidates(
        target_features,
        candidate_features,
        retrieval_scores,
    ).cpu().tolist()

    ranked = sorted(zip(retrieved, final_scores), key=lambda item: item[1], reverse=True)

    retrieved_response = [
        RetrievedCandidate(
            rank=index,
            user_id=candidate.user.user_id,
            name=candidate.user.name,
            city=candidate.user.city,
            country=candidate.user.country,
            retrieval_score=round(candidate.retrieval_score, 4),
        )
        for index, candidate in enumerate(retrieved, start=1)
    ]

    recommendation_response = [
        RecommendedUser(
            rank=index,
            user_id=candidate.user.user_id,
            name=candidate.user.name,
            city=candidate.user.city,
            country=candidate.user.country,
            retrieval_score=round(candidate.retrieval_score, 4),
            final_score=round(final_score, 4),
        )
        for index, (candidate, final_score) in enumerate(ranked[:top_k], start=1)
    ]

    if include_retrieved:
        return RecommendationResponse(
            retrieved_count=len(retrieved),
            top_k=len(recommendation_response),
            retrieved=retrieved_response,
            recommendations=recommendation_response,
        )

    return RecommendationSummaryResponse(
        retrieved_count=len(retrieved),
        top_k=len(recommendation_response),
        recommendations=recommendation_response,
    )
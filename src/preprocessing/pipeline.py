from __future__ import annotations

import logging

from src.ingestion.schema import ProcessedUser, User

from .validators import (
    calculate_age,
    parse_birth_date,
    validate_user,
)

logger = logging.getLogger(__name__)


def _normalize_text(value: str) -> str:
    return value.strip().lower()


def _normalize_interests(interests: list[str]) -> list[str]:
    cleaned = {_normalize_text(i) for i in interests if i.strip()}
    return sorted(cleaned)


def preprocess_users(
    users: list[User],
) -> list[ProcessedUser]:

    logger.info("Starting preprocessing...")

    processed_users: list[ProcessedUser] = []
    skipped_users = 0

    for user in users:

        try:
            validate_user(user)

            birth_date = parse_birth_date(user.dob)

        except ValueError as exc:
            logger.warning(str(exc))
            skipped_users += 1
            continue

        processed_users.append(
            ProcessedUser(
                user_id=user.user_id,
                name=user.name,
                gender=_normalize_text(user.gender),
                age=calculate_age(birth_date),
                interests=_normalize_interests(user.interests),
                city=_normalize_text(user.city),
                country=_normalize_text(user.country),
            )
        )

    logger.info(
        "Preprocessing completed. Processed=%d | Skipped=%d",
        len(processed_users),
        skipped_users,
    )

    return processed_users
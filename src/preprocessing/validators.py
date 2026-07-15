from __future__ import annotations

import logging
from datetime import date, datetime

from src.ingestion.schema import User

logger = logging.getLogger(__name__)


def parse_birth_date(dob: str) -> date:
    try:
        return datetime.strptime(dob, "%Y-%m-%d").date()

    except ValueError as exc:
        logger.warning("Invalid DOB format: %s", dob)
        raise ValueError(
            f"Expected DOB format YYYY-MM-DD. Received '{dob}'."
        ) from exc


def calculate_age(birth_date: date) -> int:
    today = date.today()

    return (
        today.year
        - birth_date.year
        - (
            (today.month, today.day)
            < (birth_date.month, birth_date.day)
        )
    )


def validate_user(user: User) -> None:

    if not user.user_id:
        raise ValueError("Missing UserID.")

    if not user.city:
        raise ValueError(f"User {user.user_id}: Missing city.")

    if not user.country:
        raise ValueError(f"User {user.user_id}: Missing country.")

    if not user.interests:
        raise ValueError(f"User {user.user_id}: No interests found.")
from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd

from .schema import User

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "UserID",
    "Name",
    "Gender",
    "DOB",
    "Interests",
    "City",
    "Country",
}


def _parse_interests(value: str) -> list[str]:
    if pd.isna(value):
        return []

    value = value.strip()

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, str):
            return [parsed]

        return [str(x).strip() for x in parsed]

    except Exception:
        value = value.replace("'", "")
        return [x.strip() for x in value.split(",") if x.strip()]


def load_users(csv_path: str | Path) -> list[User]:
    csv_path = Path(csv_path)

    logger.info("Loading dataset from %s", csv_path)

    if not csv_path.exists():
        logger.error("Dataset not found: %s", csv_path)
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    logger.info("Dataset loaded successfully with %d rows.", len(df))

    missing_columns = EXPECTED_COLUMNS - set(df.columns)

    if missing_columns:
        logger.error("Missing required columns: %s", missing_columns)
        raise ValueError(f"Missing columns: {missing_columns}")

    users: list[User] = []

    for row in df.itertuples(index=False):

        users.append(
            User(
                user_id=str(row.UserID),
                name=row.Name,
                gender=row.Gender,
                dob=row.DOB,
                interests=_parse_interests(row.Interests),
                city=row.City,
                country=row.Country,
            )
        )

    logger.info("Successfully created %d User objects.", len(users))

    return users
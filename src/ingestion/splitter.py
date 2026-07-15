from __future__ import annotations

import logging

from sklearn.model_selection import train_test_split

from .schema import User

logger = logging.getLogger(__name__)


def split_dataset(
    users: list[User],
    random_state: int = 42,
) -> tuple[list[User], list[User], list[User]]:
    logger.info("Splitting dataset...")

    train, temp = train_test_split(
        users,
        test_size=0.20,
        random_state=random_state,
        shuffle=True,
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=random_state,
        shuffle=True,
    )

    logger.info(
        "Dataset split completed. Train=%d | Validation=%d | Test=%d",
        len(train),
        len(val),
        len(test),
    )

    return train, val, test
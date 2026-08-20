from __future__ import annotations

import json
from pathlib import Path

import torch

from geotest import Geocoder
from src.ingestion.loader import load_users
from src.preprocessing.pipeline import preprocess_users
from src.ranking.encoder import FeatureEncoder
from src.ranking.model import TwoTowerModel
from src.retrieval.candidate_gen import CandidateGenerator

DATA_PATH = "data/processed/geouser.csv"
WORLD_CITIES_PATH = "data/raw/worldcities.csv"
MODEL_DIR = Path("models")


class AppState:

    def __init__(self) -> None:

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load processed users
        self.users = preprocess_users(
            load_users(DATA_PATH)
        )

        # Candidate retrieval
        self.generator = CandidateGenerator(self.users)

        # Geocoder (loads worldcities lookup once)
        self.geocoder = Geocoder(WORLD_CITIES_PATH)

        # Feature encoder
        self.encoder = FeatureEncoder.load(MODEL_DIR)

        # Model configuration
        with open(
            MODEL_DIR / "model_config.json",
            "r",
            encoding="utf-8",
        ) as f:
            config = json.load(f)

        self.threshold = config["calibrated_threshold"]

        # Load trained Two-Tower model
        self.model = TwoTowerModel(
            num_interests=config["num_interests"],
            num_genders=config["num_genders"],
            num_countries=config["num_countries"],
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                MODEL_DIR / "two_tower.pt",
                map_location=self.device,
            )
        )

        self.model.eval()


app_state = AppState()
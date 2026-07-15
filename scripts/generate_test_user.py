
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000/recommend/raw"
OUTPUT_PATH = Path("outputs/test_candidates.csv")

TEST_USER = {
	"UserID": "random_test_user_001",
	"Name": "jayant",
	"Gender": "male",
	"DOB": "2005-06-07",
	"Interests": "'sports', 'art'",
	"City": "Seattle",
	"Country": "United States",
}


def main() -> None:
	logger.info("POSTing test user to %s ...", API_URL)
	try:
		response = requests.post(API_URL, json=TEST_USER, timeout=10)
		response.raise_for_status()
	except requests.exceptions.ConnectionError:
		logger.error(
			"Could not connect to %s -- is the API running? Start it with: uvicorn src.api.main:app --reload",
			API_URL,
		)
		sys.exit(1)
	except requests.exceptions.HTTPError as exc:
		logger.error("API returned an error: %s", exc)
		sys.exit(1)

	data = response.json()
	recommendations = data["recommendations"]
	logger.info("Received %d recommendations for target_user_id=%s", len(recommendations), TEST_USER["UserID"])

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

	with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(["rank", "user_id", "name", "city", "country", "interests", "score"])
		writer.writerow([
			0,
			TEST_USER["UserID"],
			TEST_USER["Name"],
			TEST_USER["City"],
			TEST_USER["Country"],
			"|".join(part.strip("' ") for part in TEST_USER["Interests"].split(",")),
			"",
		])

		for rank, rec in enumerate(recommendations, start=1):
			writer.writerow([
				rank,
				rec["user_id"],
				rec["name"],
				rec["city"],
				rec["country"],
				"",
				f"{rec['final_score']:.6f}",
			])

	logger.info(
		"Wrote %d rows (1 target + %d recommendations) to %s",
		len(recommendations) + 1,
		len(recommendations),
		OUTPUT_PATH,
	)


if __name__ == "__main__":
	main()

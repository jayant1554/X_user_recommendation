# User Recommendation System

Public GitHub repository: [jayant1554/X_user_recommendation](https://github.com/jayant1554/X_user_recommendation)

This repository implements a two-stage recommendation engine for user-to-user discovery. It is organized as a modular pipeline that supports training, serving, and offline verification.

## Submission Checklist

- Public GitHub repository: [jayant1554/X_user_recommendation](https://github.com/jayant1554/X_user_recommendation)
- Stage 3 output file included in repo: [outputs/test_candidates.csv](outputs/test_candidates.csv)
- README with installation, local API run, and Stage 3 test script instructions
- README with PyTorch model/design explanation, trade-offs, and scaling notes
- README with Stage 4 architecture diagram

## What Is Included

1. Load and normalize raw user data.
2. Retrieve a small candidate pool quickly using in-memory heuristics.
3. Re-rank those candidates with a trained Two-Tower neural model.
4. Expose the pipeline through FastAPI.
5. Verify the live API with an offline integration script that writes CSV output.

The system is designed to work with the dataset schema available in this repo and to be easy to run locally from scratch.

## Project Strengths

- Modular code structure with separate ingestion, preprocessing, retrieval, ranking, API, and testing layers.
- Built-in logging support through the shared logger in [src/utils/logger.py](src/utils/logger.py).
- Clear train/validation/test separation with an 80/10/10 split.
- Fast in-memory retrieval to keep live inference lightweight.
- Saved model and encoder artifacts so the API can load trained assets at startup.
- Offline Stage 3 integration test that exercises the live API instead of bypassing it.
- Submission-ready outputs, including the generated [outputs/test_candidates.csv](outputs/test_candidates.csv) file.

## Installation

**Windows (PowerShell)**

```powershell
git clone https://github.com/jayant1554/X_user_recommendation.git
cd X_user_recommendation
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
git clone https://github.com/jayant1554/X_user_recommendation.git
cd X_user_recommendation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Download

The dataset is fetched with the Kaggle helper script.

```bash
python -m scripts.kaggle_download
```

## Train the Model

This step trains the model, evaluates it, and saves artifacts into `models/`.

```bash
python -m src.ranking.train
```

## Run the API Locally

Start the FastAPI server after training.

```bash
uvicorn src.api.main:app --reload
```
and to test use 
`http://127.0.0.1:8000/docs`

The API endpoints are:

- `POST /recommend`
- `POST /recommend/raw`

## Run Stage 3 Offline Verification

The Stage 3 script posts a dummy test user to the live API and writes the result to a CSV file.

```bash
python -m scripts.generate_test_user
```

The output is written to `outputs/test_candidates.csv`.


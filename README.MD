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

```powershell
git clone https://github.com/jayant1554/X_user_recommendation.git
cd X_user_recommendation
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Download

The dataset is fetched with the Kaggle helper script.

```powershell
python -m scripts.kaggle_download
```

## Train the Model

This step trains the model, evaluates it, and saves artifacts into `models/`.

```powershell
python -m src.ranking.train
```

## Run the API Locally

Start the FastAPI server after training.

```powershell
uvicorn src.api.main:app --reload
```

The API endpoints are:

- `POST /recommend`
- `POST /recommend/raw`

## Run Stage 3 Offline Verification

The Stage 3 script posts a dummy test user to the live API and writes the result to a CSV file.

```powershell
python -m scripts.generate_test_user
```

The output is written to `outputs/test_candidates.csv`.

## Architecture Overview

The diagram below shows the full pipeline from raw data to training, serving, and offline verification.

```mermaid
flowchart TB
	A[Raw CSV Dataset\nUserID, Name, Gender, DOB, Interests, City, Country]

	subgraph INGEST[Ingestion and Preprocessing]
		B[Ingestion Loader]
		C[Preprocessing Pipeline]
		D[Normalize Text\nParse DOB\nCompute Age]
		E[80/10/10 Split]
	end

	subgraph TRAIN[Training Pipeline]
		F[Training Split]
		G[Validation Split]
		H[Test Split]
		I[Feature Encoder Fit]
		J[Pair Sampler + Ranking Dataset]
		K[Two-Tower Training]
		L[Threshold Calibration]
		M[Offline Evaluation]
		N[Saved Model + Encoder Artifacts]
	end

	subgraph SERVE[Online Serving Path]
		O[FastAPI App Startup]
		P[Stage 1 Retrieval\nIn-memory candidate generation]
		Q[Stage 2 Neural Ranking\nPyTorch scoring]
		R[Top 10 Recommendations]
		S[Live API Response]
	end

	subgraph TEST[Offline Verification]
		T[Stage 3 Script\ngenerate_test_user.py]
		U[Live API Request]
		V[CSV Output\ntest_candidates.csv]
	end

	A --> B --> C --> D --> E
	E --> F
	E --> G
	E --> H

	F --> I --> J --> K --> N
	G --> L --> N
	H --> M

	N --> O --> P --> Q --> R --> S
	S --> T --> U --> V
```

## Design Decisions

### Why the Two-Stage Approach Is Valid

The architecture is valid because it mirrors a standard production recommender design:

- Retrieval narrows a large search space into a manageable shortlist.
- Ranking applies a more expensive neural model only to the shortlist.
- The serving layer stays lightweight and stateless.
- Training, validation, and testing are separated with an 80/10/10 split.
- The offline script exercises the real API instead of bypassing it.

This balances quality, latency, and maintainability. A monolithic model that scores every user against every other user would be too slow at scale. A purely heuristic system would be fast but less expressive.

### Stage 1 Retrieval

Stage 1 is a fast, in-memory candidate generator that uses overlapping interests plus geographic overlap from city and country to build a shortlist of up to 100 users.

- File: [src/retrieval/candidate_gen.py](src/retrieval/candidate_gen.py)

This is practical because the retrieval step does not need a heavy neural model to eliminate obviously irrelevant candidates. It is cheap, deterministic, and easy to reason about.

### Stage 2 Neural Ranking

The shortlist is then scored by a PyTorch Two-Tower model. The model encodes the target user and each candidate using interests, gender, country, and age. The final score blends the heuristic retrieval score with the neural affinity score.

- File: [src/ranking/model.py](src/ranking/model.py)

This works well because the neural model only evaluates a reduced candidate pool, which keeps inference efficient while still allowing richer learned interactions than rule-based scoring alone.

### API Layer

FastAPI exposes the recommendation service.

- `POST /recommend` accepts a processed-style payload.
- `POST /recommend/raw` accepts the raw CSV-style payload and returns only the final recommendations.

At startup, the API loads the saved model and encoder artifacts from disk. Live inference itself stays in memory and does not write to disk.

### Offline Verification

The script [scripts/generate_test_user.py](scripts/generate_test_user.py) posts a dummy test user to the live API and writes the target row plus recommendations to a CSV file.

This verifies the actual served API path rather than calling internal functions directly.

## Trade-offs and Shortcuts

The main shortcut in the current implementation is geographic proximity. The dataset only contains city and country, so the retrieval stage uses interest overlap plus same-city and same-country matching instead of true coordinate distance.

### Why Not True Geographic Proximity?

The repo does not use true geographic proximity because the dataset schema does not provide latitude or longitude values. Without coordinates, any real distance computation would require an external geocoding step, which would add extra latency, external API dependency risk, and complexity that is outside the current task scope.

Using interest overlap plus same-city and same-country matching is better here because it works directly with the existing schema, keeps retrieval fully in memory, and is deterministic and fast enough for live inference.

That choice was made because:

- there are no latitude/longitude fields in the schema
- adding geocoding would introduce extra latency and external dependencies
- the task requires working within the existing dataset structure

Other trade-offs include:

- the retrieval pool is built in memory rather than through a vector database
- the model is a compact Two-Tower architecture instead of a larger deep ranking network
- offline testing writes a CSV artifact for inspection, which is acceptable for verification but not for live inference

## Stage 4 Architectural Diagram and Production Scaling

```mermaid
flowchart LR
	U[User Traffic] --> LB[Load Balancer]
	LB --> API1[FastAPI Replica 1]
	LB --> API2[FastAPI Replica 2]
	LB --> API3[FastAPI Replica N]

	API1 --> C1[In-Memory Retrieval + Ranking]
	API2 --> C2[In-Memory Retrieval + Ranking]
	API3 --> C3[In-Memory Retrieval + Ranking]

	C1 --> M[(Shared Model Artifacts)]
	C2 --> M
	C3 --> M

	API1 --> OBS[Logs / Metrics / Traces]
	API2 --> OBS
	API3 --> OBS

	M --> CACHE[Optional Feature / Candidate Cache]
	CACHE --> API1
	CACHE --> API2
	CACHE --> API3
```

### How I Would Scale to Millions of Concurrent Requests

To handle millions of requests in production, I would change the system in several ways:

1. Run multiple stateless API replicas behind a load balancer.
2. Keep model artifacts in shared object storage or baked into the container image.
3. Add caching for repeated feature encodings and frequent candidate pools.
4. Move candidate lookup to a dedicated retrieval service or ANN index if the user base grows large.
5. Add observability with latency, error, and throughput metrics.
6. Use batch-friendly inference paths and consider GPU serving only when profiling shows it is justified.
7. Separate online serving, offline training, and offline evaluation into independent jobs.

The main limiting factors at very high scale would be memory usage, retrieval latency, and the cost of scoring a large shortlist. The current two-stage design already helps, but a production-grade system would need indexing, caching, autoscaling, and strong monitoring.

## File Map

- Ingestion: [src/ingestion/loader.py](src/ingestion/loader.py)
- Preprocessing: [src/preprocessing/pipeline.py](src/preprocessing/pipeline.py)
- Split: [src/ingestion/splitter.py](src/ingestion/splitter.py)
- Retrieval: [src/retrieval/candidate_gen.py](src/retrieval/candidate_gen.py)
- Encoder: [src/ranking/encoder.py](src/ranking/encoder.py)
- Model: [src/ranking/model.py](src/ranking/model.py)
- Training: [src/ranking/train.py](src/ranking/train.py)
- API: [src/api/main.py](src/api/main.py), [src/api/routes.py](src/api/routes.py), [src/api/services.py](src/api/services.py)
- Stage 3 script: [scripts/generate_test_user.py](scripts/generate_test_user.py)

## Summary

This repository already provides a solid recommendation-system architecture for the available dataset:

- clean data ingestion
- deterministic preprocessing
- strict train/validation/test separation
- fast in-memory retrieval
- neural re-ranking
- live API serving
- offline integration verification

The current solution is suitable for the assignment requirements and a realistic local deployment. The main remaining production gap is true geographic proximity based on coordinates. For this dataset, the project instead uses interest overlap plus same-city and same-country matching, which is a better fit for the available schema because it is fast, deterministic, and does not depend on external geocoding. Everything else is aligned with a practical production-style recommendation workflow.

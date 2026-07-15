from src.ingestion.loader import load_users
from src.preprocessing.pipeline import preprocess_users
from src.retrieval.candidate_gen import CandidateGenerator


def main():
    users = preprocess_users(
        load_users("data/raw/Assessment_TwitterDataset.csv")
    )

    generator = CandidateGenerator(users)

    target = users[0]

    candidates = generator.get_candidates(target)

    print(f"Retrieved {len(candidates)} candidates\n")

    for i, candidate in enumerate(candidates[:10], start=1):
        print(
            f"{i}. "
            f"{candidate.user.name} "
            f"| {candidate.user.city}, {candidate.user.country} "
            f"| score={candidate.retrieval_score:.4f}"
        )

if __name__ == "__main__":
    main()
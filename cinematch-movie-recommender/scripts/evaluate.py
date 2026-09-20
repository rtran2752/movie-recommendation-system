import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_movielens
from src.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CineMatch")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.65)
    args = parser.parse_args()
    movies, ratings = load_movielens(ROOT / "data" / "raw")
    metrics = evaluate(movies, ratings, args.k, args.max_users, args.alpha)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()


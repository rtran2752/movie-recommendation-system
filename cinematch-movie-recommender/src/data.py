from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED_MOVIE_COLUMNS = {"movieId", "title", "genres"}
REQUIRED_RATING_COLUMNS = {"userId", "movieId", "rating"}


def _validate(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} contains no rows")


def load_movielens(data_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    candidates = [base, base / "ml-latest-small"]
    for folder in candidates:
        movies_path, ratings_path = folder / "movies.csv", folder / "ratings.csv"
        if movies_path.exists() and ratings_path.exists():
            movies = pd.read_csv(movies_path)
            ratings = pd.read_csv(ratings_path)
            _validate(movies, REQUIRED_MOVIE_COLUMNS, "movies.csv")
            _validate(ratings, REQUIRED_RATING_COLUMNS, "ratings.csv")
            movies = movies.drop_duplicates("movieId").copy()
            ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
            ratings = ratings[ratings.movieId.isin(movies.movieId)]
            return movies, ratings
    raise FileNotFoundError(
        "MovieLens data not found. Run `python scripts/download_data.py` first."
    )


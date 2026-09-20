from __future__ import annotations

import numpy as np
import pandas as pd
from .recommender import HybridMovieRecommender


def temporal_leave_one_out(ratings: pd.DataFrame, min_ratings: int = 5):
    ordered = ratings.sort_values(["userId", "timestamp"] if "timestamp" in ratings else ["userId"])
    eligible = ordered.groupby("userId").size()
    eligible = eligible[eligible >= min_ratings].index
    subset = ordered[ordered.userId.isin(eligible)]
    test_idx = subset.groupby("userId").tail(1).index
    return ratings.drop(index=test_idx).copy(), ratings.loc[test_idx].copy()


def evaluate(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = 200,
    alpha: float = 0.65,
) -> dict[str, float]:
    train, test = temporal_leave_one_out(ratings)
    if max_users:
        keep = test.userId.drop_duplicates().head(max_users)
        test = test[test.userId.isin(keep)]
    model = HybridMovieRecommender(alpha=alpha).fit(movies, train)
    hits, reciprocal_ranks, recommended = [], [], set()
    for row in test.itertuples():
        recs = model.recommend_for_user(int(row.userId), n=k)
        ids = recs.movieId.astype(int).tolist()
        recommended.update(ids)
        if int(row.movieId) in ids:
            rank = ids.index(int(row.movieId)) + 1
            hits.append(1.0)
            reciprocal_ranks.append(1.0 / rank)
        else:
            hits.append(0.0)
            reciprocal_ranks.append(0.0)
    users = max(len(hits), 1)
    return {
        f"precision@{k}": float(np.sum(hits) / (users * k)),
        f"recall@{k}": float(np.mean(hits)) if hits else 0.0,
        f"map@{k}": float(np.mean(reciprocal_ranks)) if hits else 0.0,
        "catalog_coverage": len(recommended) / max(len(movies), 1),
        "users_evaluated": float(len(hits)),
    }


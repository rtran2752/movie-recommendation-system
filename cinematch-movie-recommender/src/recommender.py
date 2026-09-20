from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridMovieRecommender:
    """Explainable content + item-item collaborative movie recommender."""

    def __init__(self, alpha: float = 0.65, popularity_quantile: float = 0.60):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.popularity_quantile = popularity_quantile

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies = movies.drop_duplicates("movieId").reset_index(drop=True).copy()
        self.ratings = ratings[ratings.movieId.isin(self.movies.movieId)].copy()
        if self.movies.empty or self.ratings.empty:
            raise ValueError("movies and ratings must contain matching records")

        genres = self.movies.genres.fillna("").str.replace("|", " ", regex=False)
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
        self.content_matrix = self.vectorizer.fit_transform(genres)

        self.movie_ids = self.movies.movieId.to_numpy()
        self.movie_to_idx = {int(mid): i for i, mid in enumerate(self.movie_ids)}
        self.user_ids = np.sort(self.ratings.userId.unique())
        self.user_to_idx = {int(uid): i for i, uid in enumerate(self.user_ids)}

        rows = self.ratings.userId.map(self.user_to_idx).to_numpy()
        cols = self.ratings.movieId.map(self.movie_to_idx).to_numpy()
        values = self.ratings.rating.astype(float).to_numpy()
        self.user_item = csr_matrix(
            (values, (rows, cols)), shape=(len(self.user_ids), len(self.movie_ids))
        )
        self.item_similarity = cosine_similarity(self.user_item.T, dense_output=False)
        self._fit_popularity()
        return self

    def _fit_popularity(self) -> None:
        stats = self.ratings.groupby("movieId").rating.agg(["mean", "count"])
        global_mean = float(self.ratings.rating.mean())
        minimum = max(1.0, float(stats["count"].quantile(self.popularity_quantile)))
        stats["popularity"] = (
            stats["count"] * stats["mean"] + minimum * global_mean
        ) / (stats["count"] + minimum)
        self.popularity = stats["popularity"].reindex(self.movie_ids).fillna(global_mean).to_numpy()

    @staticmethod
    def _scale(values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.asarray(values, dtype=float))
        lo, hi = values.min(), values.max()
        return np.zeros_like(values) if hi <= lo else (values - lo) / (hi - lo)

    def similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        if movie_id not in self.movie_to_idx:
            return self.popular(n)
        idx = self.movie_to_idx[movie_id]
        scores = cosine_similarity(self.content_matrix[idx], self.content_matrix).ravel()
        scores[idx] = -1
        return self._result(scores, n, "content similarity")

    def recommend_for_user(self, user_id: int, n: int = 10, alpha: float | None = None) -> pd.DataFrame:
        weight = self.alpha if alpha is None else alpha
        if not 0 <= weight <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if user_id not in self.user_to_idx:
            return self.popular(n)

        row = self.user_item.getrow(self.user_to_idx[user_id])
        seen = row.indices
        if len(seen) == 0:
            return self.popular(n)

        centered = row.data - row.data.mean()
        preference_weights = centered if np.any(centered) else row.data
        collaborative = np.asarray(
            preference_weights @ self.item_similarity[seen]
        ).ravel()

        liked_mask = row.data >= max(3.5, float(row.data.mean()))
        liked = seen[liked_mask]
        if len(liked):
            profile = csr_matrix(np.asarray(self.content_matrix[liked].mean(axis=0)))
            content = cosine_similarity(profile, self.content_matrix).ravel()
        else:
            content = np.zeros(len(self.movie_ids))

        scores = (
            weight * self._scale(collaborative)
            + (1 - weight) * self._scale(content)
            + 0.05 * self._scale(self.popularity)
        )
        scores[seen] = -1
        label = f"{weight:.0%} collaborative / {1-weight:.0%} content"
        return self._result(scores, n, label)

    def popular(self, n: int = 10) -> pd.DataFrame:
        return self._result(self.popularity, n, "Bayesian popularity")

    def _result(self, scores: np.ndarray, n: int, explanation: str) -> pd.DataFrame:
        n = min(max(int(n), 1), len(scores))
        order = np.argsort(-scores)[:n]
        result = self.movies.iloc[order][["movieId", "title", "genres"]].copy()
        result["score"] = np.round(scores[order], 4)
        result["reason"] = explanation
        return result.reset_index(drop=True)

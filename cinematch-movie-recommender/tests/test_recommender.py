import pandas as pd
from src.recommender import HybridMovieRecommender
from src.evaluation import temporal_leave_one_out


def sample_data():
    movies = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": ["Space One", "Space Two", "Love One", "Love Two", "Mixed"],
        "genres": ["Sci-Fi|Adventure", "Sci-Fi|Action", "Romance", "Romance|Comedy", "Action|Comedy"],
    })
    ratings = pd.DataFrame({
        "userId": [1,1,1,2,2,2,3,3,3,3,3],
        "movieId": [1,2,3,1,2,5,1,2,3,4,5],
        "rating": [5,4,1,5,5,2,1,2,5,5,3],
        "timestamp": range(11),
    })
    return movies, ratings


def test_recommendations_exclude_seen_movies():
    movies, ratings = sample_data()
    model = HybridMovieRecommender().fit(movies, ratings)
    result = model.recommend_for_user(1, n=2)
    assert not set(result.movieId).intersection({1, 2, 3})
    assert len(result) == 2


def test_unknown_user_gets_popular_results():
    movies, ratings = sample_data()
    model = HybridMovieRecommender().fit(movies, ratings)
    result = model.recommend_for_user(999, n=3)
    assert len(result) == 3
    assert set(result.reason) == {"Bayesian popularity"}


def test_similar_movie_excludes_query_movie():
    movies, ratings = sample_data()
    model = HybridMovieRecommender().fit(movies, ratings)
    result = model.similar_movies(1, n=2)
    assert 1 not in result.movieId.tolist()


def test_temporal_split_holds_out_one_per_eligible_user():
    _, ratings = sample_data()
    train, test = temporal_leave_one_out(ratings, min_ratings=3)
    assert test.groupby("userId").size().eq(1).all()
    assert len(train) + len(test) == len(ratings)


from pathlib import Path
import streamlit as st

from src.data import load_movielens
from src.recommender import HybridMovieRecommender

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.title("🎬 CineMatch")
st.caption("An explainable hybrid movie recommendation engine")


@st.cache_data
def get_data():
    return load_movielens(ROOT / "data" / "raw")


@st.cache_resource
def get_model(_movies, _ratings):
    return HybridMovieRecommender().fit(_movies, _ratings)


try:
    movies, ratings = get_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.code("python scripts/download_data.py\nstreamlit run app.py")
    st.stop()

model = get_model(movies, ratings)
with st.sidebar:
    st.header("Controls")
    mode = st.radio("Recommendation mode", ["For a user", "Similar movies", "Popular now"])
    count = st.slider("Number of results", 5, 25, 10)
    alpha = st.slider("Collaborative weight", 0.0, 1.0, 0.65, 0.05,
                      help="Higher values prioritize patterns from community ratings.")
    st.metric("Movies", f"{len(movies):,}")
    st.metric("Ratings", f"{len(ratings):,}")

if mode == "For a user":
    user = st.selectbox("Choose a MovieLens user", model.user_ids.tolist())
    history = ratings[ratings.userId == user].nlargest(5, "rating").merge(movies, on="movieId")
    with st.expander("See this user's highly rated movies"):
        st.dataframe(history[["title", "genres", "rating"]], hide_index=True, use_container_width=True)
    results = model.recommend_for_user(int(user), count, alpha)
elif mode == "Similar movies":
    title = st.selectbox("Choose a movie", movies.title.sort_values().tolist())
    movie_id = int(movies.loc[movies.title == title, "movieId"].iloc[0])
    results = model.similar_movies(movie_id, count)
else:
    results = model.popular(count)

st.subheader("Your recommendations")
display = results.copy()
display.insert(0, "rank", range(1, len(display) + 1))
st.dataframe(display[["rank", "title", "genres", "score", "reason"]], hide_index=True, use_container_width=True)
st.caption("Scores rank titles within the current recommendation mode; they are not predicted star ratings.")


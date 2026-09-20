# CineMatch — Hybrid Movie Recommendation System

A portfolio-ready recommender built with Python, pandas, scikit-learn, and Streamlit. It combines:

- **Content-based filtering:** TF-IDF similarity over movie genres
- **Collaborative filtering:** item-item cosine similarity from user ratings
- **Hybrid ranking:** weighted combination of both signals
- **Cold-start fallback:** Bayesian popularity ranking when a user or movie is unseen
- **Evaluation:** Precision@K, Recall@K, MAP@K, coverage, and popularity baseline

## Why this is a strong data-science project

The project demonstrates data cleaning, sparse matrices, recommender-system design, model evaluation, cold-start handling, product thinking, and deployment—not merely a notebook.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
streamlit run app.py
```

MovieLens `ml-latest-small` contains about 100,000 ratings across roughly 9,700 movies. The download script retrieves it directly from GroupLens and extracts only the required CSV files.

## Run the evaluation

```bash
python scripts/evaluate.py --k 10 --max-users 200
```

The evaluation uses a temporal leave-one-out split: each eligible user's latest rating is hidden, the model trains on the earlier ratings, and the hidden movie is used as ground truth. This avoids leaking a future interaction into training.

## Test

```bash
pytest -q
```

## Project structure

```text
movie-recommender/
├── app.py                     # Streamlit product UI
├── src/
│   ├── data.py                # Validation and MovieLens loading
│   ├── recommender.py         # Content, collaborative, hybrid models
│   └── evaluation.py          # Ranking metrics and temporal split
├── scripts/
│   ├── download_data.py       # Reproducible data acquisition
│   └── evaluate.py            # Evaluation CLI
├── tests/                     # Unit tests
├── data/raw/.gitkeep
├── requirements.txt
└── .gitignore
```

## How recommendations work

For a selected user, CineMatch builds a preference profile from movies they rated. Scores are calculated as:

`hybrid_score = alpha × collaborative_score + (1 - alpha) × content_score`

Already-rated titles are removed. If insufficient collaborative history exists, the system smoothly falls back to content and Bayesian popularity. The UI exposes `alpha` so the trade-off is interpretable.

## Data and attribution

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) from GroupLens Research. Review its README and usage terms before redistributing the data. Raw data is excluded from Git.

## Next improvements

- Add matrix factorization with implicit feedback or Surprise/SVD
- Tune `alpha` on a validation set
- Add TMDB posters and plot embeddings
- Track diversity, novelty, serendipity, and subgroup performance
- Deploy to Streamlit Community Cloud

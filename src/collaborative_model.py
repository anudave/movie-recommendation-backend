import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_collaborative_model(ratings, movies):
    movie_ratings = pd.merge(ratings, movies, on='movieId')

    user_movie = movie_ratings.pivot_table(
        index='userId',
        columns='title',
        values='rating'
    ).fillna(0)

    similarity = cosine_similarity(user_movie.T)

    return pd.DataFrame(
        similarity,
        index=user_movie.columns,
        columns=user_movie.columns
    )

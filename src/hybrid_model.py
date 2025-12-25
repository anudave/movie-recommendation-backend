import pandas as pd

def hybrid_recommend(movie_title, movies, cosine_sim, indices, collab_sim, alpha=0.5, n=10):
    if movie_title not in indices or movie_title not in collab_sim:
        return "Movie not found"

    idx = indices[movie_title]
    content_scores = pd.Series(cosine_sim[idx], index=movies['title'])
    collab_scores = collab_sim[movie_title]

    final_scores = alpha * content_scores + (1 - alpha) * collab_scores
    return final_scores.sort_values(ascending=False).iloc[1:n+1]

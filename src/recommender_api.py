from src.data_loader import load_movies, load_ratings
from src.content_model import build_content_model
from src.collaborative_model import build_collaborative_model
from src.hybrid_model import hybrid_recommend

# Load CSVs
movies = load_movies('data/ml-latest-small/movies.csv')
ratings = load_ratings('data/ml-latest-small/ratings.csv')

# Build models
cosine_sim, indices = build_content_model(movies)
collab_sim = build_collaborative_model(ratings, movies)

def get_recommendations(movie_title: str):
    return hybrid_recommend(movie_title, movies, cosine_sim, indices, collab_sim)

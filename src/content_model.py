from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_content_model(movies):
    movies = movies.copy()
    movies['genres'] = movies['genres'].fillna('')
    movies['combined'] = movies['title'] + " " + movies['genres']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = {title: idx for idx, title in enumerate(movies['title'])}

    return cosine_sim, indices

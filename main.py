# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.recommender_api import get_recommendations

app = FastAPI(title="Movie Recommender API")

# Allow your frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Movie Recommender API is running"}

@app.get("/recommend")
def recommend(movie: str):
    try:
        recommendations = get_recommendations(movie)
        return {"movie": movie, "recommendations": recommendations}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Movie '{movie}' not found")

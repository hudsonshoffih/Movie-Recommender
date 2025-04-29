import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import requests
import io
from fastapi.middleware.cors import CORSMiddleware

# --- Global variables to hold processed data ---
user_movie_rating = None
user_similarity_df = None
all_movie_titles = None

# --- Data Loading and Processing Functions (adapted from notebook) ---
def load_and_process_data():
    global user_movie_rating, user_similarity_df, all_movie_titles

    print("Loading data...")
    # Load movies data
    movie_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
    try:
        s = requests.get(movie_url).content
        movies = pd.read_csv(io.StringIO(s.decode('latin-1')), sep="|", encoding="latin-1", header=None,
                             names=['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    except Exception as e:
        print(f"Error loading movie data: {e}")
        raise

    # Load ratings data
    rating_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    try:
        s = requests.get(rating_url).content
        ratings = pd.read_csv(io.StringIO(s.decode('utf-8')), sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
    except Exception as e:
        print(f"Error loading ratings data: {e}")
        raise

    print("Processing data...")
    # Merge dataframes
    df = pd.merge(movies[['movieId', 'title']], ratings, on="movieId")

    # Create user-movie rating pivot table
    user_movie_rating_pivot = df.pivot_table(index="userId", columns="title", values="rating")

    # Store all movie titles for later use if needed
    all_movie_titles = user_movie_rating_pivot.columns.tolist()

    # Fill NaN values with 0
    user_movie_rating = user_movie_rating_pivot.fillna(0)

    # Calculate user similarity
    user_similarity = cosine_similarity(user_movie_rating)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_rating.index, columns=user_movie_rating.index)

    print("Data loading and processing complete.")


def get_user_recommendation_logic(user_id, top_n=10):
    if user_movie_rating is None or user_similarity_df is None:
        raise RuntimeError("Data not loaded yet.")

    if user_id not in user_similarity_df.index:
        raise ValueError(f"User ID {user_id} not found.")

    # Get users similar to the target user (excluding the user itself)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11] # Top 10 similar users

    # Get the movie ratings of these similar users
    similar_user_movies = user_movie_rating.loc[similar_users.index]

    # Get the movies the target user has already rated (or interacted with, indicated by non-zero)
    user_movies = user_movie_rating.loc[user_id]

    # Filter for movies the target user hasn't rated (where rating is 0)
    # Consider only movies rated by similar users
    new_movies = similar_user_movies.loc[:, user_movies == 0]

    # Calculate the average rating for each potential new movie based on similar users' ratings
    # Drop movies that none of the similar users rated (all zeros in the column for this subset)
    movie_ratings = new_movies.mean(axis=0).dropna().sort_values(ascending=False)

    # Return the top N movies
    return movie_ratings.head(top_n)

# --- FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    print("Starting up FastAPI application...")
    load_and_process_data()
    yield
    # Clean up resources if needed
    print("Shutting down FastAPI application...")

# --- FastAPI App ---
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

@app.get("/")
async def read_root():
    return {"message": "Movie Recommendation API"}

@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: int, top_n: int = 10):
    try:
        recommendations = get_user_recommendation_logic(user_id, top_n)
        # Convert recommendations to a standard list format for JSON response
        response_data = [{"title": title, "predicted_rating": rating} for title, rating in recommendations.items()]
        return {"user_id": user_id, "recommendations": response_data}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- To run the app (save as main.py and run uvicorn main:app --reload) ---
# Example: uvicorn main:app --reload 
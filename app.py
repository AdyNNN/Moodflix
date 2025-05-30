from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import sqlite3
from db import init_db, get_movies_by_genre_paginated, get_user_watchlist, add_to_watchlist, remove_from_watchlist, mark_as_watched, unmark_as_watched, get_or_create_folder, rename_folder, delete_folder, move_to_folder, get_user_id_by_username, check_csv_changes
import random

# Add these imports for the content-based filtering
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Add these imports at the top of the file, after the existing imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = "supersecret"  # Change this in production

@app.before_request
def check_csv_update():
    """Check if CSV has been updated before each request"""
    if request.endpoint != 'static':  # Skip for static files
        try:
            with sqlite3.connect("moodflix.db", timeout=20) as conn:
                conn.execute("PRAGMA busy_timeout = 30000")  # Wait up to 30 seconds for locks
                if check_csv_changes(conn):
                    init_db(force_reseed=True)
                    # Clear any existing sessions to force re-login after reseed
                    session.clear()
        except sqlite3.Error as e:
            print(f"[APP] Database error during CSV check: {e}")
            # Don't raise the error to allow the request to continue

@app.route("/")
def home():
    init_db()
    # Add this line to initialize mood predictions when the app starts
    initialize_mood_predictions()
    username = session.get("username", "Guest")
    
    # Get featured movies with YouTube trailers for the header (randomly selected)
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    c.execute('''
        SELECT title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
        FROM movies
        WHERE trailer_url IS NOT NULL 
        AND trailer_url != '' 
        AND (trailer_url LIKE '%youtube.com%' OR trailer_url LIKE '%youtu.be%')
        ORDER BY RANDOM()
        LIMIT 5
    ''')
    featured_movies = []
    for row in c.fetchall():
        featured_movies.append({
            "title": row[0],
            "genre": row[1],
            "description": row[2],
            "rating": row[3],
            "release_date": row[4],
            "cast": row[5],
            "runtime": row[6],
            "poster_url": row[7],
            "trailer_url": row[8]
        })
    conn.close()
    
    # Keep track of movies we've shown to prevent duplicates
    shown_movies = set()
    
    # Add featured movie titles to shown_movies
    for movie in featured_movies:
        shown_movies.add(movie["title"])
    
    # Define all genres we want to show
    all_genres = [
        "Action", "Drama", "Thriller", "Romance", "Comedy",
        "Sci-Fi", "Horror", "Adventure", "Animation", "Fantasy"
    ]
    
    genres = {}
    for genre in all_genres:
        # Get movies for this genre, excluding ones we've already shown
        movies = get_movies_by_genre_paginated(genre, exclude_movies=list(shown_movies))
        if movies:  # Only add genre if we found movies
            genres[genre] = movies
            # Add these movies to our shown list
            for movie in movies:
                shown_movies.add(movie["title"])
    
    user_recommendations = get_user_recommendations(username)
    
    return render_template("index.html", genres=genres, username=username, featured_movies=featured_movies, user_recommendations=user_recommendations)

@app.route("/moods")
def moods():
    username = session.get("username", "Guest")
    return render_template("moods.html", username=username)

@app.route("/genre/<genre_name>/movies")
def genre_movies(genre_name):
    page = int(request.args.get("page", 0))
    movies = get_movies_by_genre_paginated(genre_name, page)
    return jsonify(movies)

@app.route("/search")
def search():
    query = request.args.get("query", "")
    username = session.get("username", "Guest")
    
    if not query:
        return render_template("search_results.html", movies=[], query="", username=username)
    
    # Check if query is a mood
    moods = ["happy", "sad", "romantic", "tense", "excited", "relaxed"]
    if query.lower() in moods:
        # Search by mood
        conn = sqlite3.connect("moodflix.db")
        c = conn.cursor()
        
        # Check if predicted_mood column exists
        c.execute("PRAGMA table_info(movies)")
        columns = [row[1] for row in c.fetchall()]
        
        if "predicted_mood" in columns:
            # Use predicted mood
            c.execute("""
                SELECT title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
                FROM movies
                WHERE predicted_mood = ?
                ORDER BY rating DESC
                LIMIT 50
            """, (query.lower(),))
        else:
            # Fall back to genre-based mood mapping
            mood_to_genres = {
                "happy": ["Comedy", "Animation", "Family"],
                "sad": ["Drama"],
                "romantic": ["Romance"],
                "tense": ["Thriller", "Horror", "Mystery"],
                "excited": ["Action", "Adventure", "Sci-Fi"],
                "relaxed": ["Drama", "Comedy", "Family"]
            }
            
            genres = mood_to_genres.get(query.lower(), ["Drama"])
            genre_conditions = " OR ".join([f"genre LIKE '%{genre}%'" for genre in genres])
            
            c.execute(f"""
                SELECT title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
                FROM movies
                WHERE ({genre_conditions})
                ORDER BY rating DESC
                LIMIT 50
            """)
    else:
        # Regular search in title, genre, cast, and description
        conn = sqlite3.connect("moodflix.db")
        c = conn.cursor()
        c.execute('''
            SELECT title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
            FROM movies
            WHERE title LIKE ? OR genre LIKE ? OR "cast" LIKE ? OR description LIKE ?
            ORDER BY rating DESC
            LIMIT 50
        ''', (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
    
    movies = []
    for row in c.fetchall():
        movies.append({
            "title": row[0],
            "genre": row[1],
            "description": row[2],
            "rating": row[3],
            "release_date": row[4],
            "cast": row[5],
            "runtime": row[6],
            "poster_url": row[7] or "/static/posters/placeholder.jpg",
            "trailer_url": row[8]
        })
    
    conn.close()
    
    return render_template("search_results.html", movies=movies, query=query, username=username)

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("moodflix.db")
        c = conn.cursor()
        
        # First check if username exists
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if not user:
            return "Invalid username", 401
        elif user[0] != password:
            return "Invalid password", 401
        
        session["username"] = username
        return redirect(url_for("home"))

    return render_template("signin.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("moodflix.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            return "Username already exists", 400
        finally:
            conn.close()

        session["username"] = username
        return redirect(url_for("home"))

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

@app.route("/profile")
def profile():
    username = session.get("username", "Guest")
    if username == "Guest":
        return redirect(url_for("signin"))
    return render_template("profile.html", username=username)

# Function to get similar movies based on content-based filtering
def get_similar_movies(movie_title, limit=5):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import re
    
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    
    # Get the current movie's details
    c.execute('''
        SELECT id, title, genre, "cast", description, rating, release_date
        FROM movies
        WHERE title = ?
    ''', (movie_title,))
    
    current_movie = c.fetchone()
    if not current_movie:
        conn.close()
        return []
    
    current_id, current_title, current_genre, current_cast, current_description, current_rating, current_release_date = current_movie
    
    # Get all other movies
    c.execute('''
        SELECT id, title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
        FROM movies
        WHERE id != ? AND rating >= ?
    ''', (current_id, max(0, current_rating - 2)))  # Get movies with similar or better rating
    
    all_movies = c.fetchall()
    conn.close()
    
    if not all_movies:
        return []
    
    # Create a list of all movies including the current one for vectorization
    movie_data = [(current_id, current_title, current_genre, current_cast, current_description, current_rating, current_release_date)] + [
        (movie[0], movie[1], movie[2], movie[6], movie[3], movie[4], movie[5]) for movie in all_movies
    ]
    
    # Clean and prepare text data
    def clean_text(text):
        if not text:
            return ""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Create content strings for each movie by combining features
    content_strings = []
    for _, title, genre, cast, desc, rating, release_date in movie_data:
        # Clean and weight the features
        clean_genre = clean_text(genre)
        clean_cast = clean_text(cast)
        clean_desc = clean_text(desc)
        
        # Extract year from release date
        year = release_date[:4] if release_date else ""
        
        # Weight features differently
        weighted_content = (
            f"{clean_genre} {clean_genre} {clean_genre} {clean_genre} " +  # Genre has highest weight
            f"{clean_cast} {clean_cast} {clean_cast} " +                   # Cast has high weight
            f"{year} {year} " +                                           # Year has medium weight
            f"{clean_desc}"                                               # Description has lowest weight
        )
        
        content_strings.append(weighted_content)
    
    # Create TF-IDF vectors with improved parameters
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),  # Consider both unigrams and bigrams
        min_df=2,            # Ignore terms that appear in less than 2 documents
        max_df=0.95          # Ignore terms that appear in more than 95% of documents
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(content_strings)
    
    # Calculate cosine similarity between the current movie and all other movies
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get indices of movies sorted by similarity score
    similar_indices = cosine_sim.argsort()[::-1]
    
    # Get more movies than needed for randomization
    top_similar_indices = similar_indices[:min(limit * 3, len(similar_indices))]
    
    # Randomly select from top similar movies
    selected_indices = np.random.choice(
        top_similar_indices,
        size=min(limit, len(top_similar_indices)),
        replace=False
    )
    
    # Get the selected similar movies
    similar_movies = []
    for idx in selected_indices:
        movie = all_movies[idx]
        similar_movies.append({
            "title": movie[1],
            "genre": movie[2],
            "description": movie[3],
            "rating": movie[4],
            "release_date": movie[5],
            "cast": movie[6],
            "runtime": movie[7],
            "poster_url": movie[8] or "/static/posters/placeholder.jpg",
            "trailer_url": movie[9],
            "similarity_score": float(cosine_sim[idx])
        })
    
    return similar_movies

@app.route("/movie/<title>")
def view_movie(title):
    import sqlite3
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()

    # Use exact title match — assuming titles are unique
    c.execute('''
        SELECT title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
        FROM movies
        WHERE title = ?
    ''', (title,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "Movie not found", 404

    movie = {
        "title": row[0],
        "genre": row[1],
        "description": row[2],
        "rating": row[3],
        "release_date": row[4],
        "cast": row[5],
        "runtime": row[6],
        "poster_url": row[7],
        "trailer_url": row[8]
    }

    # Check if movie is in user's watchlist
    username = session.get("username", "Guest")
    in_watchlist = False
    is_watched = False
    
    if username != "Guest":
        conn = sqlite3.connect("moodflix.db")
        c = conn.cursor()
        user_id = get_user_id_by_username(username)
        
        if user_id:
            # Check watchlist
            c.execute('''
                SELECT 1 FROM watchlist w
                JOIN movies m ON w.movie_id = m.id
                WHERE w.user_id = ? AND m.title = ?
            ''', (user_id, title))
            in_watchlist = c.fetchone() is not None
            
            # Check watched status
            c.execute('''
                SELECT 1 FROM watched w
                JOIN movies m ON w.movie_id = m.id
                WHERE w.user_id = ? AND m.title = ?
            ''', (user_id, title))
            is_watched = c.fetchone() is not None
        
        conn.close()
    
    # Get similar movies
    similar_movies = get_similar_movies(title)

    return render_template("view_movies.html", movie=movie, username=username, 
                          in_watchlist=in_watchlist, is_watched=is_watched,
                          similar_movies=similar_movies)

@app.route("/api/similar-movies/<title>")
def api_similar_movies(title):
    similar_movies = get_similar_movies(title)
    return jsonify(similar_movies)

@app.route("/explore")
def explore():
    username = session.get("username", "Guest")
    return render_template("explore.html", username=username)

# New routes for watchlist management

@app.route("/watchlist")
def watchlist():
    username = session.get("username", "Guest")
    if username == "Guest":
        return redirect(url_for("signin"))
    
    watchlist_data = get_user_watchlist(username)
    return render_template("watchlist.html", username=username, watchlist=watchlist_data)

@app.route("/api/watchlist/get")
def api_get_watchlist():
    if session.get("username", "Guest") == "Guest":
        return jsonify({"success": False, "message": "Please sign in first"}), 401
    
    watchlist_data = get_user_watchlist(session["username"])
    return jsonify({"success": True, "watchlist": watchlist_data})

@app.route("/api/watchlist/add", methods=["POST"])
def api_add_to_watchlist():
    if session.get("username", "Guest") == "Guest":
        return jsonify({"success": False, "message": "Please sign in first"}), 401
    
    data = request.json
    movie_title = data.get("movie_title")
    folder_name = data.get("folder_name")
    
    if not movie_title:
        return jsonify({"success": False, "message": "Movie title is required"}), 400
    
    success = add_to_watchlist(session["username"], movie_title, folder_name)
    
    if success:
        return jsonify({"success": True, "message": "Movie added to watchlist"})
    else:
        return jsonify({"success": False, "message": "Failed to add movie to watchlist"}), 500

@app.route("/api/watchlist/remove", methods=["POST"])
def api_remove_from_watchlist():
    if session.get("username", "Guest") == "Guest":
        return jsonify({"success": False, "message": "Please sign in first"}), 401
    
    data = request.json
    movie_title = data.get("movie_title")
    
    if not movie_title:
        return jsonify({"success": False, "message": "Movie title is required"}), 400
    
    success = remove_from_watchlist(session["username"], movie_title)
    
    if success:
        return jsonify({"success": True, "message": "Movie removed from watchlist"})
    else:
        return jsonify({"success": False, "message": "Failed to remove movie from watchlist"}), 500

@app.route("/api/watchlist/watched", methods=["POST"])
def api_mark_as_watched():
    if session.get("username", "Guest") == "Guest":
        return jsonify({"success": False, "message": "Please sign in first"}), 401
    
    data = request.json
    movie_title = data.get("movie_title")
    rating = data.get("rating", 0)
    
    if not movie_title:
        return jsonify({"success": False, "message": "Movie title is required"}), 400
    
    success = mark_as_watched(session["username"], movie_title, rating)
    
    if success:
        return jsonify({"success": True, "message": "Movie marked as watched"})
    else:
        return jsonify({"success": False, "message": "Failed to mark movie as watched"}), 500

@app.route("/api/watchlist/unwatched", methods=["POST"])
def api_unmark_as_watched():
    if session.get("username", "Guest") == "Guest":
        return jsonify({"success": False, "message": "Please sign in first"}), 401
    
    data = request.json
    movie_title = data.get("movie_title")
    
    if not movie_title:
        return jsonify({"success": False, "message": "Movie title is required"}), 400
    
    success = unmark_as_watched(session["username"], movie_title)
    
    if success:
        return jsonify({"success": True, "message": "Movie marked as unwatched"})
    else:
        return jsonify({"success": False, "message": "Failed to mark movie as unwatched"}), 500

@app.route("/api/folder/create", methods=["POST"])
def api_create_folder():
    if session.get("username", "Guest") == "Guest":
        return jsonify({"success": False, "message": "Please sign in first"}), 401
    
    data = request.json
    folder_name = data.get("folder_name")
    
    if not folder_name:
        return jsonify({"success": False, "message": "Folder name is required"}), 400
    
    user_id = get_user_id_by_username(session["username"])
    folder_id = get_or_create_folder(user_id, folder_name)
    
    if folder_id:
        return jsonify({"success": True, "message": "Folder created", "folder_id": folder_id})
    else:
        return jsonify({"success": False, "message": "Failed to create folder"}), 500

@app.route("/api/folder/rename", methods=["POST"])
def api_rename_folder():
    if session.get("username", "Guest") == "Guest":
        return jsonify({"success": False, "message": "Please sign in first"}), 401
    
    data = request.json
    old_name = data.get("old_name")
    new_name = data.get("new_name")
    
    if not old_name or not new_name:
        return jsonify({"success": False, "message": "Both old and new folder names are required"}), 400
    
    user_id = get_user_id_by_username(session["username"])
    success = rename_folder(user_id, old_name, new_name)
    
    if success:
        return jsonify({"success": True, "message": "Folder renamed"})
    else:
        return jsonify({"success": False, "message": "Failed to rename folder"}), 500

@app.route("/api/folder/delete", methods=["POST"])
def api_delete_folder():
    if session.get("username", "Guest") == "Guest":
        return jsonify({"success": False, "message": "Please sign in first"}), 401
    
    data = request.json
    folder_name = data.get("folder_name")
    
    if not folder_name:
        return jsonify({"success": False, "message": "Folder name is required"}), 400
    
    user_id = get_user_id_by_username(session["username"])
    success = delete_folder(user_id, folder_name)
    
    if success:
        return jsonify({"success": True, "message": "Folder deleted"})
    else:
        return jsonify({"success": False, "message": "Failed to delete folder"}), 500

@app.route("/api/watchlist/move", methods=["POST"])
def api_move_to_folder():
    if session.get("username", "Guest") == "Guest":
        return jsonify({"success": False, "message": "Please sign in first"}), 401
    
    data = request.json
    movie_title = data.get("movie_title")
    folder_name = data.get("folder_name")
    
    if not movie_title:
        return jsonify({"success": False, "message": "Movie title is required"}), 400
    
    success = move_to_folder(session["username"], movie_title, folder_name)
    
    if success:
        return jsonify({"success": True, "message": "Movie moved to folder"})
    else:
        return jsonify({"success": False, "message": "Failed to move movie to folder"}), 500

@app.route("/api/movies/by_mood", methods=["GET"])
def get_movies_by_mood():
    mood = request.args.get("mood", "")
    if not mood:
        return jsonify({"error": "Mood parameter is required"}), 400
    
    # Map moods to genres and other attributes
    mood_mappings = {
        "happy": ["Comedy", "Animation", "Family"],
        "sad": ["Drama", "Romance"],
        "romantic": ["Romance", "Drama"],
        "tense": ["Thriller", "Horror", "Mystery"],
        "excited": ["Action", "Adventure", "Sci-Fi"],
        "relaxed": ["Drama", "Comedy", "Family"]
    }
    
    # Get genres for the selected mood
    genres = mood_mappings.get(mood.lower(), ["Drama"])
    
    # Build a query to find movies matching the mood
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    
    # Use OR conditions for multiple genres
    genre_conditions = " OR ".join([f"genre LIKE '%{genre}%'" for genre in genres])
    
    # Get movies matching the mood with a good rating (>= 6.5)
    c.execute(f"""
        SELECT title, genre, description, rating, poster_url, trailer_url
        FROM movies
        WHERE ({genre_conditions}) AND rating >= 6.5
        ORDER BY RANDOM()
        LIMIT 20
    """)
    
    movies = []
    for row in c.fetchall():
        movies.append({
            "title": row[0],
            "genre": row[1],
            "description": row[2],
            "rating": row[3],
            "poster_url": row[4] or "/static/posters/placeholder.jpg",
            "trailer_url": row[5]
        })
    
    conn.close()
    return jsonify(movies)

# Add these new routes and functions after the existing /api/movies/by_mood route
@app.route("/train_mood_classifier")
def train_mood_classifier():
    """Admin route to train the mood classifier model"""
    # Only allow admins to train the model
    if session.get("username", "Guest") != "admin":
        return "Unauthorized", 401
    
    # Get movie data from database
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    c.execute("""
        SELECT description, genre, rating, release_date, runtime 
        FROM movies 
        WHERE description IS NOT NULL 
        AND description != ''
        AND genre IS NOT NULL
    """)
    data = c.fetchall()
    conn.close()
    
    if len(data) < 50:  # Ensure we have enough data
        return "Not enough movie data to train model", 400
    
    descriptions = [row[0] for row in data]
    genres = [row[1] for row in data]
    ratings = [row[2] for row in data]
    release_dates = [row[3] for row in data]
    runtimes = [row[4] for row in data]
    
    # Enhanced mood mapping with weighted genres and combinations
    mood_mapping = {
        "happy": {
            "primary_genres": ["Comedy", "Animation", "Family"],
            "secondary_genres": ["Adventure", "Fantasy"],
            "genre_combinations": [
                ("Comedy", "Romance"),  # Romantic comedies
                ("Comedy", "Adventure"),  # Fun adventures
                ("Animation", "Family")   # Family animations
            ],
            "attributes": {
                "min_rating": 6.5,  # Happy movies should be well-rated
                "preferred_runtime": (90, 120)  # Not too long, not too short
            }
        },
        "sad": {
            "primary_genres": ["Drama"],
            "secondary_genres": ["Romance", "War"],
            "genre_combinations": [
                ("Drama", "Romance"),  # Romantic dramas
                ("Drama", "War"),      # War dramas
                ("Drama", "History")   # Historical dramas
            ],
            "attributes": {
                "min_rating": 7.0,  # Sad movies should be impactful
                "preferred_runtime": (100, 160)  # Can be longer
            }
        },
        "romantic": {
            "primary_genres": ["Romance"],
            "secondary_genres": ["Drama", "Comedy"],
            "genre_combinations": [
                ("Romance", "Comedy"),  # Rom-coms
                ("Romance", "Drama"),   # Romantic dramas
                ("Romance", "Fantasy")  # Fantasy romance
            ],
            "attributes": {
                "min_rating": 6.0,
                "preferred_runtime": (90, 130)
            }
        },
        "tense": {
            "primary_genres": ["Thriller", "Horror", "Mystery"],
            "secondary_genres": ["Crime", "Sci-Fi"],
            "genre_combinations": [
                ("Thriller", "Mystery"),  # Mystery thrillers
                ("Horror", "Thriller"),   # Horror thrillers
                ("Crime", "Mystery")      # Crime mysteries
            ],
            "attributes": {
                "min_rating": 6.5,
                "preferred_runtime": (90, 150)
            }
        },
        "excited": {
            "primary_genres": ["Action", "Adventure", "Sci-Fi"],
            "secondary_genres": ["Fantasy", "Thriller"],
            "genre_combinations": [
                ("Action", "Sci-Fi"),     # Sci-fi action
                ("Adventure", "Fantasy"),  # Fantasy adventures
                ("Action", "Adventure")    # Action adventures
            ],
            "attributes": {
                "min_rating": 6.0,
                "preferred_runtime": (100, 160)  # Can be longer for epics
            }
        },
        "relaxed": {
            "primary_genres": ["Drama", "Comedy", "Family"],
            "secondary_genres": ["Documentary", "Music"],
            "genre_combinations": [
                ("Drama", "Comedy"),    # Light dramas
                ("Family", "Fantasy"),  # Family fantasy
                ("Comedy", "Music")     # Musical comedies
            ],
            "attributes": {
                "min_rating": 6.0,
                "preferred_runtime": (90, 120)  # Not too demanding
            }
        }
    }
    
    # Function to determine mood score based on genres and attributes
    def get_mood_scores(movie_genres, rating, runtime):
        scores = {mood: 0.0 for mood in mood_mapping.keys()}
        movie_genres = set(g.strip() for g in movie_genres.split('|'))
        
        for mood, criteria in mood_mapping.items():
            score = 0.0
            
            # Primary genres (highest weight)
            primary_matches = len(set(criteria["primary_genres"]) & movie_genres)
            score += primary_matches * 0.4
            
            # Secondary genres (medium weight)
            secondary_matches = len(set(criteria["secondary_genres"]) & movie_genres)
            score += secondary_matches * 0.2
            
            # Genre combinations (high weight)
            for genre1, genre2 in criteria["genre_combinations"]:
                if genre1 in movie_genres and genre2 in movie_genres:
                    score += 0.3
            
            # Rating criteria
            if rating >= criteria["attributes"]["min_rating"]:
                score += 0.1
            
            # Runtime criteria
            min_runtime, max_runtime = criteria["attributes"]["preferred_runtime"]
            if min_runtime <= runtime <= max_runtime:
                score += 0.1
            
            scores[mood] = score
        
        return scores
    
    # Generate mood labels based on the enhanced criteria
    moods = []
    for i in range(len(data)):
        scores = get_mood_scores(genres[i], ratings[i], runtimes[i])
        primary_mood = max(scores.items(), key=lambda x: x[1])[0]
        moods.append(primary_mood)
    
    # Create and train the model
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(descriptions)
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, moods, test_size=0.2, random_state=42)
    
    # Train RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Save the model and vectorizer
    with open('mood_classifier.pkl', 'wb') as f:
        pickle.dump((clf, vectorizer), f)
    
    # Test accuracy
    accuracy = clf.score(X_test, y_test)
    
    # Update movie database with predicted moods
    update_movie_moods()
    
    return f"Model trained successfully with accuracy: {accuracy:.2f}"

def update_movie_moods():
    """Update the database with predicted moods for all movies"""
    # Check if model exists
    if not os.path.exists('mood_classifier.pkl'):
        return False
    
    # Load the model
    with open('mood_classifier.pkl', 'rb') as f:
        clf, vectorizer = pickle.load(f)
    
    # Get all movies
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    
    # Check if mood column exists, if not add it
    c.execute("PRAGMA table_info(movies)")
    columns = [row[1] for row in c.fetchall()]
    if "predicted_mood" not in columns:
        c.execute("ALTER TABLE movies ADD COLUMN predicted_mood TEXT")
    
    # Get movies without predicted moods
    c.execute("SELECT id, description FROM movies WHERE description IS NOT NULL AND description != ''")
    movies = c.fetchall()
    
    # Predict moods and update database
    for movie_id, description in movies:
        if description:
            # Predict mood
            X = vectorizer.transform([description])
            predicted_mood = clf.predict(X)[0]
            
            # Update database
            c.execute("UPDATE movies SET predicted_mood = ? WHERE id = ?", (predicted_mood, movie_id))
    
    conn.commit()
    conn.close()
    return True

@app.route("/api/movies/by_predicted_mood")
def get_movies_by_predicted_mood():
    """API endpoint to get movies by their predicted mood"""
    mood = request.args.get("mood", "")
    if not mood:
        return jsonify({"error": "Mood parameter is required"}), 400
    
    # Get movies with the predicted mood
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    
    # Check if predicted_mood column exists
    c.execute("PRAGMA table_info(movies)")
    columns = [row[1] for row in c.fetchall()]
    
    if "predicted_mood" not in columns:
        # If column doesn't exist, fall back to genre-based mood selection
        return get_movies_by_mood()
    
    # Get movies with the predicted mood
    c.execute("""
        SELECT title, genre, description, rating, poster_url, trailer_url
        FROM movies
        WHERE predicted_mood = ? AND rating >= 6.0
        ORDER BY RANDOM()
        LIMIT 20
    """, (mood,))
    
    movies = []
    for row in c.fetchall():
        movies.append({
            "title": row[0],
            "genre": row[1],
            "description": row[2],
            "rating": row[3],
            "poster_url": row[4] or "/static/posters/placeholder.jpg",
            "trailer_url": row[5]
        })
    
    conn.close()
    
    # If not enough movies found with predicted mood, supplement with genre-based
    if len(movies) < 10:
        genre_based_movies = get_movies_by_mood().json
        # Add non-duplicate movies
        existing_titles = {movie["title"] for movie in movies}
        for movie in genre_based_movies:
            if movie["title"] not in existing_titles and len(movies) < 20:
                movies.append(movie)
                existing_titles.add(movie["title"])
    
    return jsonify(movies)

@app.route("/api/movies/by_multiple_moods")
def get_movies_by_multiple_moods():
    """API endpoint to get movies by multiple predicted moods"""
    moods_param = request.args.get("moods", "")
    page = int(request.args.get("page", 1))
    sort_by = request.args.get("sort_by", "title")
    sort_order = request.args.get("order", "asc")
    page_size = 20
    
    if not moods_param:
        return jsonify({"error": "Moods parameter is required"}), 400
    
    # Parse the comma-separated moods
    selected_moods = [mood.strip().lower() for mood in moods_param.split(',')]
    
    if not selected_moods:
        return jsonify({"error": "At least one mood is required"}), 400
    
    # Get movies with any of the predicted moods
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    
    # Map moods to genres for mood-based recommendations
    mood_to_genres = {
        "happy": ["Comedy", "Animation", "Family"],
        "sad": ["Drama"],
        "romantic": ["Romance"],
        "tense": ["Thriller", "Horror", "Mystery"],
        "excited": ["Action", "Adventure", "Sci-Fi"],
        "relaxed": ["Drama", "Comedy", "Family"]
    }
    
    # Collect all genres for selected moods
    all_genres = []
    for mood in selected_moods:
        all_genres.extend(mood_to_genres.get(mood, []))
    
    # Remove duplicates while preserving order
    unique_genres = list(dict.fromkeys(all_genres))
    
    # Calculate offset for pagination
    offset = (page - 1) * page_size
    
    if unique_genres:
        # Create genre conditions
        genre_conditions = " OR ".join([f"genre LIKE ?" for _ in unique_genres])
        genre_params = [f'%{genre}%' for genre in unique_genres]
        
        # First, get total count
        count_query = f"""
            SELECT COUNT(DISTINCT title)
            FROM movies
            WHERE ({genre_conditions}) AND rating >= 6.0
        """
        c.execute(count_query, genre_params)
        total_count = c.fetchone()[0]
        
        # Add sorting
        sort_clause = f"ORDER BY {sort_by} {sort_order}"
        if sort_by == "year":
            sort_clause = f"ORDER BY CAST(substr(release_date, 1, 4) AS INTEGER) {sort_order}"
        
        # Then get paginated results
        query = f"""
            SELECT DISTINCT title, genre, description, rating, poster_url, trailer_url, release_date
            FROM movies
            WHERE ({genre_conditions}) AND rating >= 6.0
            {sort_clause}
            LIMIT ? OFFSET ?
        """
        params = genre_params + [page_size, offset]
        
        c.execute(query, params)
        
        movies = []
        for row in c.fetchall():
            movies.append({
                "title": row[0],
                "genre": row[1],
                "description": row[2],
                "rating": row[3],
                "poster_url": row[4] or "/static/posters/placeholder.jpg",
                "trailer_url": row[5],
                "release_date": row[6],
                "matched_mood": "genre-based"
            })
        
        conn.close()
        
        response_data = {
            "movies": movies,
            "total_count": total_count,
            "page": page,
            "has_more": total_count > (page * page_size)
        }
        return jsonify(response_data)
    
    return jsonify({
        "movies": [],
        "total_count": 0,
        "page": page,
        "has_more": False
    })

def initialize_mood_predictions():
    """Initialize mood predictions for all movies if not already done."""
    # Check if the mood classifier model exists
    if not os.path.exists('mood_classifier.pkl'):
        print("Mood classifier model not found. Training...")
        # If the model doesn't exist, train it
        # Note: You might want to restrict this to an admin route in production
        train_mood_classifier()
    
    # Check if the predicted_mood column exists in the movies table
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    c.execute("PRAGMA table_info(movies)")
    columns = [row[1] for row in c.fetchall()]
    conn.close()
    
    if "predicted_mood" not in columns:
        print("predicted_mood column not found. Creating and populating...")
        # If the column doesn't exist, add it and populate it
        update_movie_moods()
    else:
        print("predicted_mood column found. Checking for unpredicted movies...")
        # Check if there are any movies with NULL predicted_mood
        conn = sqlite3.connect("moodflix.db")
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM movies WHERE predicted_mood IS NULL")
        count = c.fetchone()[0]
        conn.close()
        
        if count > 0:
            print(f"Found {count} movies with NULL predicted_mood. Updating...")
            update_movie_moods()
        else:
            print("All movies have predicted moods.")

# Add this route to your app.py file
@app.route('/api/movies/by_multiple_moods_sorted')
def get_sorted_movies_by_moods():
    moods = request.args.get('moods', '').split(',')
    sort_by = request.args.get('sort_by', 'title')
    order = request.args.get('order', 'desc')
    
    # Clean up moods list
    moods = [mood.strip() for mood in moods if mood.strip()]
    
    if not moods:
        return jsonify([])
    
    # Get sorted movies from database
    from db import get_movies_by_multiple_moods_sorted
    movies = get_movies_by_multiple_moods_sorted(moods, sort_by, order)
    
    return jsonify(movies)

def get_user_recommendations(username, limit=5):
    if username == "Guest":
        return []
        
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    user_id = get_user_id_by_username(username)
    
    if not user_id:
        conn.close()
        return []
    
    # Get genres from user's watchlist and watched movies
    c.execute("""
        SELECT m.genre, w.user_rating
        FROM movies m
        LEFT JOIN watchlist wl ON m.id = wl.movie_id
        LEFT JOIN watched w ON m.id = w.movie_id AND w.user_id = ?
        WHERE (wl.user_id = ? OR w.user_id = ?)
    """, (user_id, user_id, user_id))
    
    user_genres = []
    genre_weights = {}
    
    for row in c.fetchall():
        if not row[0]:  # Skip if genre is None
            continue
            
        # Split genres and process each one
        genres = [g.strip() for g in row[0].split('|')]
        user_rating = row[1] if row[1] else 0
        
        for genre in genres:
            if genre:
                user_genres.append(genre)
                # Weight genres by user rating (if available)
                weight = 1 + (user_rating / 10) if user_rating > 0 else 1
                genre_weights[genre] = genre_weights.get(genre, 0) + weight
    
    # If user has no preferences yet, return empty list instead of default recommendations
    if not user_genres:
        conn.close()
        return []
    
    # Get most frequent genres weighted by user ratings
    top_genres = sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)[:3]
    top_genres = [genre for genre, _ in top_genres]
    
    # Create weighted conditions for each genre
    genre_conditions = []
    genre_params = []
    for i, genre in enumerate(top_genres):
        # Give higher weight to primary genres
        weight = 3 - i  # 3 for first genre, 2 for second, 1 for third
        genre_conditions.append(f"(CASE WHEN m.genre LIKE ? THEN {weight} ELSE 0 END)")
        genre_params.append(f"%{genre}%")
    
    # Combine conditions into a relevance score
    relevance_score = " + ".join(genre_conditions)
    
    query = f"""
        SELECT DISTINCT m.title, m.genre, m.description, m.rating, 
               m.release_date, m."cast", m.runtime, m.poster_url, m.trailer_url,
               ({relevance_score}) as relevance
        FROM movies m
        WHERE m.rating >= 7.0
        AND m.id NOT IN (
            SELECT movie_id FROM watchlist WHERE user_id = ?
            UNION
            SELECT movie_id FROM watched WHERE user_id = ?
        )
        AND (
            {' OR '.join(['m.genre LIKE ?' for _ in top_genres])}
        )
        ORDER BY relevance DESC, m.rating DESC
        LIMIT ?
    """
    
    c.execute(query, genre_params + genre_params + [user_id, user_id, limit])
    
    recommendations = []
    for row in c.fetchall():
        recommendations.append({
            "title": row[0],
            "genre": row[1],
            "description": row[2],
            "rating": row[3],
            "release_date": row[4],
            "cast": row[5],
            "runtime": row[6],
            "poster_url": row[7] or "/static/posters/placeholder.jpg",
            "trailer_url": row[8]
        })
    
    conn.close()
    return recommendations

@app.route("/api/user-recommendations")
def api_user_recommendations():
    username = session.get("username", "Guest")
    page = int(request.args.get("page", 0))
    limit = 5  # Number of movies per page
    offset = page * limit
    
    if username == "Guest":
        return jsonify([])
        
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    user_id = get_user_id_by_username(username)
    
    if not user_id:
        conn.close()
        return jsonify([])
    
    # Get genres from user's watchlist and watched movies
    c.execute("""
        SELECT m.genre, w.user_rating
        FROM movies m
        LEFT JOIN watchlist wl ON m.id = wl.movie_id
        LEFT JOIN watched w ON m.id = w.movie_id AND w.user_id = ?
        WHERE (wl.user_id = ? OR w.user_id = ?)
    """, (user_id, user_id, user_id))
    
    user_genres = []
    genre_weights = {}
    
    for row in c.fetchall():
        if not row[0]:  # Skip if genre is None
            continue
            
        # Split genres and process each one
        genres = [g.strip() for g in row[0].split('|')]
        user_rating = row[1] if row[1] else 0
        
        for genre in genres:
            if genre:
                user_genres.append(genre)
                # Weight genres by user rating (if available)
                weight = 1 + (user_rating / 10) if user_rating > 0 else 1
                genre_weights[genre] = genre_weights.get(genre, 0) + weight
    
    # If user has no preferences yet, return empty list instead of default recommendations
    if not user_genres:
        conn.close()
        return jsonify([])
    
    # Get most frequent genres weighted by user ratings
    top_genres = sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)[:3]
    top_genres = [genre for genre, _ in top_genres]
    
    # Create weighted conditions for each genre
    genre_conditions = []
    genre_params = []
    for i, genre in enumerate(top_genres):
        # Give higher weight to primary genres
        weight = 3 - i  # 3 for first genre, 2 for second, 1 for third
        genre_conditions.append(f"(CASE WHEN m.genre LIKE ? THEN {weight} ELSE 0 END)")
        genre_params.append(f"%{genre}%")
    
    # Combine conditions into a relevance score
    relevance_score = " + ".join(genre_conditions)
    
    query = f"""
        SELECT DISTINCT m.title, m.genre, m.description, m.rating, 
               m.release_date, m."cast", m.runtime, m.poster_url, m.trailer_url,
               ({relevance_score}) as relevance
        FROM movies m
        WHERE m.rating >= 7.0
        AND m.id NOT IN (
            SELECT movie_id FROM watchlist WHERE user_id = ?
            UNION
            SELECT movie_id FROM watched WHERE user_id = ?
        )
        AND (
            {' OR '.join(['m.genre LIKE ?' for _ in top_genres])}
        )
        ORDER BY relevance DESC, m.rating DESC
        LIMIT ? OFFSET ?
    """
    
    c.execute(query, genre_params + genre_params + [user_id, user_id, limit, offset])
    
    recommendations = []
    for row in c.fetchall():
        recommendations.append({
            "title": row[0],
            "genre": row[1],
            "description": row[2],
            "rating": row[3],
            "release_date": row[4],
            "cast": row[5],
            "runtime": row[6],
            "poster_url": row[7] or "/static/posters/placeholder.jpg",
            "trailer_url": row[8]
        })
    
    conn.close()
    return jsonify(recommendations)

@app.route("/api/movies/by_genres", methods=['POST'])
def get_movies_by_genres():
    data = request.json
    genres = data.get('genres', [])
    page = data.get('page', 1)
    page_size = 20
    offset = (page - 1) * page_size
    
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    
    if genres:
        # Create genre conditions
        genre_conditions = " OR ".join([f"genre LIKE ?" for _ in genres])
        genre_params = [f'%{genre}%' for genre in genres]
        
        # Get total count
        count_query = f"""
            SELECT COUNT(DISTINCT title)
            FROM movies
            WHERE ({genre_conditions})
        """
        c.execute(count_query, genre_params)
        total_count = c.fetchone()[0]
        
        # Get paginated results
        query = f"""
            SELECT DISTINCT title, genre, description, rating, poster_url, trailer_url
            FROM movies
            WHERE ({genre_conditions})
            ORDER BY rating DESC
            LIMIT ? OFFSET ?
        """
        params = genre_params + [page_size, offset]
        
        c.execute(query, params)
        
        movies = []
        for row in c.fetchall():
            movies.append({
                "title": row[0],
                "genre": row[1],
                "description": row[2],
                "rating": row[3],
                "poster_url": row[4] or "/static/posters/placeholder.jpg",
                "trailer_url": row[5]
            })
    else:
        # If no genres selected, get all movies
        c.execute("""
            SELECT COUNT(DISTINCT title) FROM movies
        """)
        total_count = c.fetchone()[0]
        
        c.execute("""
            SELECT DISTINCT title, genre, description, rating, poster_url, trailer_url
            FROM movies
            ORDER BY rating DESC
            LIMIT ? OFFSET ?
        """, (page_size, offset))
        
        movies = []
        for row in c.fetchall():
            movies.append({
                "title": row[0],
                "genre": row[1],
                "description": row[2],
                "rating": row[3],
                "poster_url": row[4] or "/static/posters/placeholder.jpg",
                "trailer_url": row[5]
            })
    
    conn.close()
    
    return jsonify({
        "movies": movies,
        "total_count": total_count,
        "page": page,
        "has_more": total_count > (page * page_size)
    })

if __name__ == "__main__":
    app.run(debug=True)
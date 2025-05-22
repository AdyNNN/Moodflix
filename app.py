from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import sqlite3
from db import init_db, get_movies_by_genre_paginated, get_user_watchlist, add_to_watchlist, remove_from_watchlist, mark_as_watched, unmark_as_watched, get_or_create_folder, rename_folder, delete_folder, move_to_folder, get_user_id_by_username
import random

# Add these imports for the content-based filtering
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
app.secret_key = "supersecret"  # Change this in production

@app.route("/")
def home():
    init_db()
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
    
    genres = {
        "Action": get_movies_by_genre_paginated("Action"),
        "Drama": get_movies_by_genre_paginated("Drama"),
        "Thriller": get_movies_by_genre_paginated("Thriller"),
        "Romance": get_movies_by_genre_paginated("Romance"),
        "Comedy": get_movies_by_genre_paginated("Comedy"),
    }
    return render_template("index.html", genres=genres, username=username, featured_movies=featured_movies)

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
    
    # Search for movies matching the query
    conn = sqlite3.connect("moodflix.db")
    c = conn.cursor()
    
    # Search in title, genre, cast, and description
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
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["username"] = username
            return redirect(url_for("home"))
        else:
            return "Invalid credentials", 401

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
        SELECT id, title, genre, "cast", description
        FROM movies
        WHERE title = ?
    ''', (movie_title,))
    
    current_movie = c.fetchone()
    if not current_movie:
        conn.close()
        return []
    
    current_id, current_title, current_genre, current_cast, current_description = current_movie
    
    # Get all other movies
    c.execute('''
        SELECT id, title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
        FROM movies
        WHERE id != ?
    ''', (current_id,))
    
    all_movies = c.fetchall()
    conn.close()
    
    if not all_movies:
        return []
    
    # Create a list of all movies including the current one for vectorization
    movie_data = [(current_id, current_title, current_genre, current_cast, current_description)] + [
        (movie[0], movie[1], movie[2], movie[6], movie[3]) for movie in all_movies
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
    for _, title, genre, cast, desc in movie_data:
        # Clean and weight the features
        clean_genre = clean_text(genre)
        clean_cast = clean_text(cast)
        clean_desc = clean_text(desc)
        
        # Weight features differently (genre and cast are more important than description)
        # Repeat important features to give them more weight
        weighted_content = (
            f"{clean_genre} {clean_genre} {clean_genre} " +  # Genre has highest weight
            f"{clean_cast} {clean_cast} " +                  # Cast has medium weight
            f"{clean_desc}"                                  # Description has lowest weight
        )
        
        content_strings.append(weighted_content)
    
    # Create TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(content_strings)
    
    # Calculate cosine similarity between the current movie and all other movies
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get indices of movies sorted by similarity score
    similar_indices = cosine_sim.argsort()[::-1]
    
    # Get the top similar movies
    similar_movies = []
    for idx in similar_indices[:limit]:
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
            "similarity_score": float(cosine_sim[idx])  # Convert numpy float to Python float
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

if __name__ == "__main__":
    app.run(debug=True)

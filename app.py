from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import sqlite3
from db import init_db, get_movies_by_genre_paginated, get_user_watchlist, add_to_watchlist, remove_from_watchlist, mark_as_watched, unmark_as_watched, get_or_create_folder, rename_folder, delete_folder, move_to_folder, get_user_id_by_username

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

    return render_template("view_movies.html", movie=movie, username=username, 
                          in_watchlist=in_watchlist, is_watched=is_watched)

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

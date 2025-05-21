import sqlite3
import os
import csv
import random

DB_FILE = 'moodflix.db'
CSV_FILE = 'movies.csv'

def init_db(force_reseed=False):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()

        # --- Create tables ---
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                genre TEXT,
                description TEXT,
                rating REAL
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                folder_id INTEGER,
                UNIQUE(user_id, movie_id),
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(movie_id) REFERENCES movies(id),
                FOREIGN KEY(folder_id) REFERENCES folders(id)
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS watched (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                date_watched TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_rating INTEGER DEFAULT 0,
                UNIQUE(user_id, movie_id),
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(movie_id) REFERENCES movies(id)
            )
        ''')
        
        # New table for folders
        c.execute('''
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, name),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')

        # --- Add any missing columns ---
        c.execute("PRAGMA table_info(movies)")
        existing_columns = [row[1] for row in c.fetchall()]
        new_columns = {
            "description": "TEXT",
            "release_date": "TEXT",
            "cast": "TEXT",
            "runtime": "INTEGER",
            "poster_url": "TEXT",
            "trailer_url": "TEXT"
        }
        for column, datatype in new_columns.items():
            if column not in existing_columns:
                print(f"[DB] Adding column '{column}' to movies...")
                c.execute(f"ALTER TABLE movies ADD COLUMN {column} {datatype}")

        # --- Add missing columns to watchlist ---
        c.execute("PRAGMA table_info(watchlist)")
        existing_columns = [row[1] for row in c.fetchall()]
        if "date_added" not in existing_columns:
            c.execute("ALTER TABLE watchlist ADD COLUMN date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        if "folder_id" not in existing_columns:
            c.execute("ALTER TABLE watchlist ADD COLUMN folder_id INTEGER REFERENCES folders(id)")

        # --- Add missing columns to watched ---
        c.execute("PRAGMA table_info(watched)")
        existing_columns = [row[1] for row in c.fetchall()]
        if "date_watched" not in existing_columns:
            c.execute("ALTER TABLE watched ADD COLUMN date_watched TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        if "user_rating" not in existing_columns:
            c.execute("ALTER TABLE watched ADD COLUMN user_rating INTEGER DEFAULT 0")

        # ✅ Optional: force reseed (DEV ONLY)
        if force_reseed:
            print("[DB] Force clearing existing movies...")
            c.execute("DELETE FROM movies")

        # --- Seed if empty ---
        c.execute("SELECT COUNT(*) FROM movies")
        if c.fetchone()[0] == 0 and os.path.exists(CSV_FILE):
            print("[DB] Seeding movies from CSV...")
            with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    title = row.get("title", "").strip()
                    genre = row.get("genres", "").strip()
                    description = row.get("overview", "").strip()
                    try:
                        rating = round(float(row.get("rating", 0)), 1)
                    except:
                        rating = 0
                    release_date = row.get("release_date", "").strip() or "Unknown"
                    cast = row.get("cast", "").strip() or "Unknown"
                    try:
                        runtime = int(float(row.get("runtime", 0)))
                    except:
                        runtime = 0
                    poster_url = row.get("poster_url", "").strip() or "/static/posters/placeholder.jpg"
                    trailer_url = row.get("trailer_url", "").strip()

                    c.execute('''
                        INSERT INTO movies (title, genre, description, rating, release_date, cast, runtime, poster_url, trailer_url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (title, genre, description, rating, release_date, cast, runtime, poster_url, trailer_url))
            print("[DB] Seeding complete.")

        conn.commit()

def get_movies_by_genre(genre, limit=5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
    SELECT title, genre, description, rating, release_date, "cast", runtime, poster_url
    FROM movies
    WHERE genre LIKE ? LIMIT ?
''', (f'%{genre}%', limit))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "title": row[0],
            "genre": row[1],
            "description": row[2],
            "rating": row[3],
            "release_date": row[4],
            "cast": row[5],  # exposed to template as 'actors'
            "runtime": row[6],
            "poster_url": row[7] or "/static/posters/placeholder.jpg"
        }
        for row in rows
    ]

def get_random_movies_by_genre(genre, limit=5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
        FROM movies
        WHERE genre LIKE ?
    ''', (f'%{genre}%',))
    rows = c.fetchall()
    conn.close()

    # Randomly pick `limit` number of movies
    selected = random.sample(rows, min(limit, len(rows)))

    return [
        {
            "title": row[0],
            "genre": row[1],
            "description": row[2],
            "rating": row[3],
            "release_date": row[4],
            "cast": row[5],
            "runtime": row[6],
            "poster_url": row[7] or "/static/posters/placeholder.jpg",
            "trailer_url": row[8]
        }
        for row in selected
    ]

def get_movies_by_genre_paginated(genre, page=0, limit=5):
    offset = page * limit
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
        FROM movies
        WHERE genre LIKE ?
        LIMIT ? OFFSET ?
    ''', (f'%{genre}%', limit, offset))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "title": row[0],
            "genre": row[1],
            "description": row[2],
            "rating": row[3],
            "release_date": row[4],
            "cast": row[5],
            "runtime": row[6],
            "poster_url": row[7] or "/static/posters/placeholder.jpg",
            "trailer_url": row[8]
        }
        for row in rows
    ]

# New functions for watchlist management

def get_user_id_by_username(username):
    """Get user ID from username"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def get_movie_id_by_title(title):
    """Get movie ID from title"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM movies WHERE title = ?", (title,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def add_to_watchlist(username, movie_title, folder_name=None):
    """Add a movie to user's watchlist"""
    user_id = get_user_id_by_username(username)
    movie_id = get_movie_id_by_title(movie_title)
    
    if not user_id or not movie_id:
        return False
    
    folder_id = None
    if folder_name:
        folder_id = get_or_create_folder(user_id, folder_name)
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT OR REPLACE INTO watchlist (user_id, movie_id, folder_id) VALUES (?, ?, ?)",
            (user_id, movie_id, folder_id)
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def remove_from_watchlist(username, movie_title):
    """Remove a movie from user's watchlist"""
    user_id = get_user_id_by_username(username)
    movie_id = get_movie_id_by_title(movie_title)
    
    if not user_id or not movie_id:
        return False
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND movie_id = ?",
            (user_id, movie_id)
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def mark_as_watched(username, movie_title, rating=0):
    """Mark a movie as watched with optional rating"""
    user_id = get_user_id_by_username(username)
    movie_id = get_movie_id_by_title(movie_title)
    
    if not user_id or not movie_id:
        return False
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT OR REPLACE INTO watched (user_id, movie_id, user_rating) VALUES (?, ?, ?)",
            (user_id, movie_id, rating)
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def unmark_as_watched(username, movie_title):
    """Remove watched status from a movie"""
    user_id = get_user_id_by_username(username)
    movie_id = get_movie_id_by_title(movie_title)
    
    if not user_id or not movie_id:
        return False
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            "DELETE FROM watched WHERE user_id = ? AND movie_id = ?",
            (user_id, movie_id)
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def get_or_create_folder(user_id, folder_name):
    """Get folder ID or create if it doesn't exist"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Check if folder exists
    c.execute("SELECT id FROM folders WHERE user_id = ? AND name = ?", (user_id, folder_name))
    result = c.fetchone()
    
    if result:
        folder_id = result[0]
    else:
        # Create new folder
        c.execute(
            "INSERT INTO folders (user_id, name) VALUES (?, ?)",
            (user_id, folder_name)
        )
        conn.commit()
        folder_id = c.lastrowid
    
    conn.close()
    return folder_id

def rename_folder(user_id, old_name, new_name):
    """Rename a folder"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            "UPDATE folders SET name = ? WHERE user_id = ? AND name = ?",
            (new_name, user_id, old_name)
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def delete_folder(user_id, folder_name):
    """Delete a folder and move its movies to 'No Folder'"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Get folder ID
    c.execute("SELECT id FROM folders WHERE user_id = ? AND name = ?", (user_id, folder_name))
    result = c.fetchone()
    
    if not result:
        conn.close()
        return False
    
    folder_id = result[0]
    
    try:
        # Set folder_id to NULL for all movies in this folder
        c.execute("UPDATE watchlist SET folder_id = NULL WHERE user_id = ? AND folder_id = ?", (user_id, folder_id))
        
        # Delete the folder
        c.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def get_user_watchlist(username):
    """Get all movies in user's watchlist with folder and watched info"""
    user_id = get_user_id_by_username(username)
    
    if not user_id:
        return []
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Get all watchlist items with movie details, folder info, and watched status
    c.execute('''
        SELECT 
            m.id, m.title, m.genre, m.poster_url, m.rating, 
            f.name as folder_name,
            CASE WHEN w.id IS NOT NULL THEN 1 ELSE 0 END as is_watched,
            w.user_rating
        FROM 
            watchlist wl
            JOIN movies m ON wl.movie_id = m.id
            LEFT JOIN folders f ON wl.folder_id = f.id
            LEFT JOIN watched w ON wl.user_id = w.user_id AND wl.movie_id = w.movie_id
        WHERE 
            wl.user_id = ?
        ORDER BY
            f.name, m.title
    ''', (user_id,))
    
    rows = c.fetchall()
    
    # Get all folders for this user
    c.execute("SELECT id, name FROM folders WHERE user_id = ? ORDER BY name", (user_id,))
    folders = [{"id": row[0], "name": row[1]} for row in c.fetchall()]
    
    conn.close()
    
    # Organize movies by folder
    movies_by_folder = {}
    for row in rows:
        movie = {
            "id": row[0],
            "title": row[1],
            "genre": row[2],
            "poster_url": row[3],
            "rating": row[4],
            "is_watched": row[6],
            "user_rating": row[7] or 0
        }
        
        folder_name = row[5] or "Uncategorized"
        
        if folder_name not in movies_by_folder:
            movies_by_folder[folder_name] = []
        
        movies_by_folder[folder_name].append(movie)
    
    return {
        "folders": folders,
        "movies_by_folder": movies_by_folder
    }

def move_to_folder(username, movie_title, folder_name):
    """Move a movie to a different folder"""
    user_id = get_user_id_by_username(username)
    movie_id = get_movie_id_by_title(movie_title)
    
    if not user_id or not movie_id:
        return False
    
    folder_id = None
    if folder_name and folder_name != "Uncategorized":
        folder_id = get_or_create_folder(user_id, folder_name)
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            "UPDATE watchlist SET folder_id = ? WHERE user_id = ? AND movie_id = ?",
            (folder_id, user_id, movie_id)
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()

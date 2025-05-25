import sqlite3
import os
import csv
import random
import hashlib

DB_FILE = 'moodflix.db'
CSV_FILE = 'movies.csv'

def get_csv_hash():
    """Calculate hash of CSV file to detect changes"""
    if not os.path.exists(CSV_FILE):
        return None
    
    with open(CSV_FILE, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def check_csv_changes(conn):
    """Check if CSV file has changed by comparing stored and current hash"""
    c = conn.cursor()
    
    # Create table to store CSV hash if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Get current CSV hash
    current_hash = get_csv_hash()
    if not current_hash:
        return False
    
    # Get stored hash
    c.execute("SELECT value FROM app_state WHERE key = 'csv_hash'")
    result = c.fetchone()
    stored_hash = result[0] if result else None
    
    # Update stored hash if CSV has changed
    if current_hash != stored_hash:
        c.execute("INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)", 
                 ('csv_hash', current_hash))
        conn.commit()
        return True
    
    return False

def init_db(force_reseed=False):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()

        # --- Create tables ---
        # 1. Users table (must be first as other tables reference it)
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')

        # 2. Movies table
        c.execute('''
            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                genre TEXT,
                description TEXT,
                rating REAL,
                release_date TEXT,
                cast TEXT,
                runtime INTEGER,
                poster_url TEXT,
                trailer_url TEXT,
                predicted_mood TEXT
            )
        ''')

        # 3. Folders table (references users)
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

        # 4. Watchlist table (references users, movies, and folders)
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

        # 5. Watched table (references users and movies)
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

        # Check if we need to reseed the database
        need_reseed = force_reseed
        
        if not need_reseed and os.path.exists(CSV_FILE):
            # Check if CSV has changed by comparing hashes
            if check_csv_changes(conn):
                print("[DB] CSV file has changed. Reseeding database...")
                need_reseed = True
        
        if need_reseed:
            # Clear existing movies and their related data
            print("[DB] Clearing existing movies and related data...")
            c.execute("DELETE FROM watchlist")  # Clear watchlist first due to foreign key
            c.execute("DELETE FROM watched")    # Clear watched data
            c.execute("DELETE FROM movies")     # Then clear movies
            
            # Delete the mood classifier model to force retraining
            if os.path.exists('mood_classifier.pkl'):
                os.remove('mood_classifier.pkl')
                print("[DB] Removed old mood classifier model for retraining...")
            
            # Reseed from CSV
            if os.path.exists(CSV_FILE):
                print("[DB] Seeding movies from CSV...")
                with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    
                    # Map CSV columns to database columns
                    column_mapping = {
                        'title': 'title',
                        'genres': 'genre',  # Note the difference
                        'overview': 'description',  # Note the difference
                        'rating': 'rating',
                        'release_date': 'release_date',
                        'cast': 'cast',
                        'runtime': 'runtime',
                        'poster_url': 'poster_url',
                        'trailer_url': 'trailer_url'
                    }
                    
                    for row in reader:
                        # Extract and clean data using the mapping
                        data = {}
                        for csv_col, db_col in column_mapping.items():
                            value = row.get(csv_col, '').strip()
                            
                            # Special handling for different column types
                            if db_col == 'rating':
                                try:
                                    value = round(float(value or 0), 1)
                                except:
                                    value = 0
                            elif db_col == 'runtime':
                                try:
                                    value = int(float(value or 0))
                                except:
                                    value = 0
                            elif db_col == 'poster_url':
                                value = value or '/static/posters/placeholder.jpg'
                                # Ensure poster URL starts with /static/posters/
                                if not value.startswith('/static/posters/'):
                                    # Check if the file exists in the posters directory
                                    poster_path = os.path.join('static/posters', os.path.basename(value))
                                    if os.path.exists(poster_path):
                                        value = f'/static/posters/{os.path.basename(value)}'
                                    else:
                                        value = '/static/posters/placeholder.jpg'
                            
                            data[db_col] = value
                        
                        # Insert into database
                        c.execute('''
                            INSERT INTO movies (
                                title, genre, description, rating, 
                                release_date, cast, runtime, 
                                poster_url, trailer_url
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            data['title'], data['genre'], data['description'],
                            data['rating'], data['release_date'], data['cast'],
                            data['runtime'], data['poster_url'], data['trailer_url']
                        ))
                
                print("[DB] Seeding complete.")
                
                # Initialize mood predictions for the new data
                initialize_mood_predictions()
            else:
                print("[DB] Warning: movies.csv not found!")

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

# Add this function to the db.py file, after the existing functions

def initialize_mood_predictions():
    """Initialize the predicted_mood column in the movies table"""
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        
        # Check if predicted_mood column exists
        c.execute("PRAGMA table_info(movies)")
        columns = [row[1] for row in c.fetchall()]
        
        if "predicted_mood" not in columns:
            print("[DB] Adding predicted_mood column to movies table...")
            c.execute("ALTER TABLE movies ADD COLUMN predicted_mood TEXT")
            conn.commit()
            
            # Try to load and use the mood classifier if it exists
            try:
                import pickle
                from sklearn.feature_extraction.text import TfidfVectorizer
                import os
                
                if os.path.exists('mood_classifier.pkl'):
                    print("[DB] Found mood classifier model, predicting moods...")
                    with open('mood_classifier.pkl', 'rb') as f:
                        clf, vectorizer = pickle.load(f)
                    
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
                    print("[DB] Mood predictions complete.")
                else:
                    print("[DB] No mood classifier model found. Run /train_mood_classifier to create one.")
            except Exception as e:
                print(f"[DB] Error initializing mood predictions: {e}")
                # If prediction fails, use genre-based mood mapping as fallback
                print("[DB] Using genre-based mood mapping as fallback...")
                
                # Simple genre to mood mapping
                mood_mapping = {
                    "Comedy": "happy",
                    "Animation": "happy",
                    "Family": "happy",
                    "Drama": "sad",
                    "Romance": "romantic",
                    "Thriller": "tense",
                    "Horror": "tense",
                    "Mystery": "tense",
                    "Action": "excited",
                    "Adventure": "excited",
                    "Sci-Fi": "excited"
                }
                
                c.execute("SELECT id, genre FROM movies")
                movies = c.fetchall()
                
                for movie_id, genre_list in movies:
                    if genre_list:
                        primary_genre = genre_list.split('|')[0].strip()
                        mood = mood_mapping.get(primary_genre, "relaxed")  # Default to relaxed
                        c.execute("UPDATE movies SET predicted_mood = ? WHERE id = ?", (mood, movie_id))
                
                conn.commit()
                print("[DB] Fallback mood mapping complete.")
# Add these functions to your db.py file

def quick_sort_movies(movies, sort_by="title", order="desc"):
    """
    Sort movies using the QuickSort algorithm.
    
    Args:
        movies: List of movie dictionaries
        sort_by: Field to sort by (title, rating, year)
        order: Sort order - "asc" for ascending, "desc" for descending
    
    Returns:
        Sorted list of movies
    """
    if not movies:
        return []
    
    # Create a copy to avoid modifying the original list
    movies_copy = movies.copy()
    
    # Define compare functions for different sort options
    def compare_by_title(a, b):
        a_title = a.get('title', '').lower()
        b_title = b.get('title', '').lower()
        if order == "asc":
            return a_title < b_title
        else:
            return a_title > b_title
    
    def compare_by_rating(a, b):
        a_rating = float(a.get('rating', 0))
        b_rating = float(b.get('rating', 0))
        if order == "asc":
            return a_rating < b_rating
        else:
            return a_rating > b_rating
    
    def compare_by_year(a, b):
        # Extract year from release_date or use a default year
        try:
            a_year = int(a.get('release_date', '').split('-')[0]) if a.get('release_date') else 0
        except (IndexError, ValueError):
            a_year = 0
            
        try:
            b_year = int(b.get('release_date', '').split('-')[0]) if b.get('release_date') else 0
        except (IndexError, ValueError):
            b_year = 0
            
        if order == "asc":
            return a_year < b_year
        else:
            return a_year > b_year
    
    # Select the appropriate compare function
    if sort_by == "title":
        compare_func = compare_by_title
    elif sort_by == "rating":
        compare_func = compare_by_rating
    elif sort_by == "year":
        compare_func = compare_by_year
    else:
        compare_func = compare_by_title  # Default to title
    
    # QuickSort implementation
    def _quick_sort(arr, low, high):
        if low < high:
            pivot_index = _partition(arr, low, high)
            _quick_sort(arr, low, pivot_index - 1)
            _quick_sort(arr, pivot_index + 1, high)
    
    def _partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if compare_func(arr[j], pivot):
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    # Execute the sort
    _quick_sort(movies_copy, 0, len(movies_copy) - 1)
    return movies_copy


def get_movies_by_multiple_moods_sorted(moods, sort_by="title", order="desc", limit=20):
    """
    Get movies by multiple moods and sort them using QuickSort.
    
    Args:
        moods: List of mood strings
        sort_by: Field to sort by (title, rating, year)
        order: Sort order - "asc" for ascending, "desc" for descending
        limit: Maximum number of movies to return
    
    Returns:
        Sorted list of movies matching the moods
    """
    if not moods:
        return []
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Create placeholders for the IN clause
    placeholders = ','.join(['?' for _ in moods])
    
    # First check if predicted_mood column exists
    c.execute("PRAGMA table_info(movies)")
    columns = [row[1] for row in c.fetchall()]
    
    movies = []
    
    if "predicted_mood" in columns:
        # Query movies that match any of the provided moods
        query = f'''
            SELECT DISTINCT title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
            FROM movies
            WHERE predicted_mood IN ({placeholders})
            AND title IS NOT NULL
            AND title != ''
        '''
        
        c.execute(query, moods)
        rows = c.fetchall()
        
        # Convert to list of dictionaries
        movies = [
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
    
    # If not enough movies found with predicted moods, supplement with genre-based
    if len(movies) < 15:
        # Map moods to genres for fallback
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
        for mood in moods:
            all_genres.extend(mood_to_genres.get(mood, []))
        
        # Remove duplicates while preserving order
        unique_genres = list(dict.fromkeys(all_genres))
        
        if unique_genres:
            # Create genre conditions
            genre_conditions = " OR ".join([f"genre LIKE ?" for _ in unique_genres])
            genre_params = [f'%{genre}%' for genre in unique_genres]
            
            # Get existing movie titles to avoid duplicates
            existing_titles = {movie["title"] for movie in movies}
            
            if existing_titles:
                title_conditions = " AND ".join([f"title != ?" for _ in existing_titles])
                query = f"""
                    SELECT title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
                    FROM movies
                    WHERE ({genre_conditions}) AND ({title_conditions})
                    AND title IS NOT NULL AND title != ''
                """
                params = genre_params + list(existing_titles)
            else:
                query = f"""
                    SELECT title, genre, description, rating, release_date, "cast", runtime, poster_url, trailer_url
                    FROM movies
                    WHERE ({genre_conditions})
                    AND title IS NOT NULL AND title != ''
                """
                params = genre_params
            
            c.execute(query, params)
            
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
    
    # Sort using QuickSort
    sorted_movies = quick_sort_movies(movies, sort_by, order)
    
    # Apply limit
    return sorted_movies[:limit]


def get_movies_by_mood_sorted(mood, sort_by="title", order="desc", limit=20):
    """
    Get movies by a single mood and sort them using QuickSort.
    
    Args:
        mood: Single mood string
        sort_by: Field to sort by (title, rating, year)
        order: Sort order - "asc" for ascending, "desc" for descending
        limit: Maximum number of movies to return
    
    Returns:
        Sorted list of movies matching the mood
    """
    return get_movies_by_multiple_moods_sorted([mood], sort_by, order, limit)


# Update your existing get_movies_by_multiple_moods function to use sorting
def get_movies_by_multiple_moods(moods, limit=20):
    """
    Get movies by multiple moods (backward compatibility).
    This function maintains compatibility with existing code.
    """
    return get_movies_by_multiple_moods_sorted(moods, "rating", "desc", limit)
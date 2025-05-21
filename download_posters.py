import csv
import os
import requests

# === CONFIGURATION ===
TMDB_API_KEY = "5047b2668891813cd433394e1123058f"  # 🔑 Replace with your real API key
CSV_INPUT = "movies.csv"  # Use your existing CSV file
POSTER_DIR = "static/posters"  # Local folder to store images
os.makedirs(POSTER_DIR, exist_ok=True)


def get_tmdb_poster_url(title):
    """Search TMDB for a movie and return the poster URL (or None)."""
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if results and results[0].get("poster_path"):
            return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
    except Exception as e:
        print(f"[ERROR] {title}: {e}")
    return None


def download_poster(url, title):
    """Download and save the poster locally, return the relative path."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        safe_title = title.lower().replace(" ", "_").replace(":", "").replace("/", "_")
        filename = f"{safe_title}.jpg"
        filepath = os.path.join(POSTER_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return f"/static/posters/{filename}"
    except Exception as e:
        print(f"[ERROR] Failed to download poster for {title}: {e}")
        return "/static/placeholder.jpg"


def main():
    with open(CSV_INPUT, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    for row in rows:
        title = row.get("title", "").strip()
        if not title:
            continue

        print(f"[INFO] Processing: {title}")
        poster_url = get_tmdb_poster_url(title)
        if poster_url:
            local_path = download_poster(poster_url, title)
        else:
            local_path = "/static/placeholder.jpg"

        row["poster_url"] = local_path

    # Overwrite original CSV (or you can write to a new one)
    fieldnames = reader.fieldnames + (["poster_url"] if "poster_url" not in reader.fieldnames else [])
    with open("movies_with_posters.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n✅ Done! Posters saved to /static/posters/, CSV updated as movies_with_posters.csv")


if __name__ == "__main__":
    main()

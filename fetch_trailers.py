import pandas as pd
import requests
import time

# Your TMDb API key
API_KEY = ''  # <-- Replace with your actual API key
BASE_URL = 'https://api.themoviedb.org/3'

# Load your movie dataset (must contain a 'title' column)
df = pd.read_csv('movies.csv')
print("Loaded CSV:", df.shape)

# Helper: Get TMDb movie ID from title
def get_movie_id(title):
    url = f'{BASE_URL}/search/movie'
    params = {'api_key': API_KEY, 'query': title}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get('results')
        if results:
            return results[0].get('id')
    return None

# Helper: Get YouTube embed trailer URL from movie ID
def get_trailer_url(movie_id):
    url = f'{BASE_URL}/movie/{movie_id}/videos'
    params = {'api_key': API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get('results', [])
        for video in results:
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                return f"https://www.youtube.com/embed/{video['key']}"
    return None

trailer_urls = []

# Loop through movies and collect trailer URLs
for title in df['title']:
    print(f"Searching trailer for: {title}")
    movie_id = get_movie_id(title)
    if movie_id:
        trailer_url = get_trailer_url(movie_id)
        print(f"Found trailer: {trailer_url}")
        trailer_urls.append(trailer_url)
    else:
        print("No movie ID found")
        trailer_urls.append(None)
    time.sleep(0.3)

# Add new column and save to new CSV
df['trailer_url'] = trailer_urls
df.to_csv('movies_with_trailers.csv', index=False)

print("✅ Done! Saved as 'movies_with_trailers.csv'")

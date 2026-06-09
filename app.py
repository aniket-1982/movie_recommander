import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re

st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

TMDB_API_KEY = "f33d667edec047a1736cc90539cee400"

# ── Fetch poster ──
def fetch_poster(movie_title):
    try:
        clean_title = re.sub(r'\(\d{4}\)', '', movie_title).strip()
        if ', The' in clean_title:
            clean_title = 'The ' + clean_title.replace(', The', '')
        if ', A ' in clean_title:
            clean_title = 'A ' + clean_title.replace(', A ', ' ')
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": clean_title}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            vote = data['results'][0].get('vote_average', 0)
            year = data['results'][0].get('release_date', '')[:4]
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}", vote, year
    except:
        pass
    return None, 0, ''

# ── Fetch popular movies ──
@st.cache_data
def fetch_popular_movies():
    movies_list = []
    for page in range(1, 6):
        url = "https://api.themoviedb.org/3/movie/popular"
        params = {"api_key": TMDB_API_KEY, "page": page}
        res = requests.get(url, params=params, timeout=5).json()
        movies_list.extend(res.get('results', []))
    return movies_list

# ── Load dataset ──
@st.cache_resource
def load_data():
    movies = pd.read_csv('movies.csv')
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies = movies[(movies['year'] >= 2010) & (movies['year'] <= 2026)]
    movies = movies.reset_index(drop=True)
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return movies, cosine_sim

# ── Display ALL movies in grid ──
def display_movies(movie_df, cols_count=5):
    if movie_df.empty:
        st.warning("No movies found for selected genres!")
        return
    st.markdown(f"**{len(movie_df)} movies found!**")
    cols = st.columns(cols_count)
    for i, row in enumerate(movie_df.itertuples()):
        poster, vote, year = fetch_poster(row.title)
        with cols[i % cols_count]:
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.image("https://placehold.co/200x300?text=No+Poster", use_container_width=True)
            st.markdown(f"**{row.title}**")
            st.caption(f"⭐ {round(vote,1)} | 📅 {year}")
            st.caption(f"🎭 {row.genres[:25]}..

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import requests

# ── Page config ──
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

TMDB_API_KEY = "f33d667edec047a1736cc90539cee400"

# ── Fetch poster from TMDB ──
def fetch_poster(movie_title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": movie_title}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            overview = data['results'][0].get('overview', 'No description available.')
            vote = data['results'][0].get('vote_average', 0)
            year = data['results'][0].get('release_date', '')[:4]
            if poster_path:
                return (
                    f"https://image.tmdb.org/t/p/w500{poster_path}",
                    overview, vote, year
                )
    except:
        pass
    return None, 'No description available.', 0, ''

# ── Fetch latest popular movies from TMDB ──
@st.cache_data
def fetch_poster(movie_title):
    try:
        # Clean the title — remove year like "(1995)" from the end
        import re
        clean_title = re.sub(r'\(\d{4}\)', '', movie_title).strip()
        # Also handle titles like "The, Matrix" → "The Matrix"
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
            overview = data['results'][0].get('overview', 'No description available.')
            vote = data['results'][0].get('vote_average', 0)
            year = data['results'][0].get('release_date', '')[:4]
            if poster_path:
                return (
                    f"https://image.tmdb.org/t/p/w500{poster_path}",
                    overview, vote, year
                )
    except:
        pass
    return None, 'No description available.', 0, ''

# ── Load & train ──
@st.cache_resource
def load_and_train():
    movies  = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')

    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    tfidf        = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
    cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices      = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    reader   = Reader(rating_scale=(0.5, 5.0))
    data     = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    svd      = SVD(n_factors=100, random_state=42)
    svd.fit(trainset)

    return movies, ratings, cosine_sim, indices, svd

# ── Hybrid function ──
def hybrid_recommend(user_id, title, movies, cosine_sim, indices, svd, num=10):
    if title not in indices:
        return None
    idx        = indices[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:50]
    candidates = movies.iloc[[i[0] for i in sim_scores]][['movieId','title','genres']].copy()
    candidates['content_score'] = [i[1] for i in sim_scores]
    candidates['collab_score']  = candidates['movieId'].apply(
        lambda x: svd.predict(user_id, x).est)
    mn, mx = candidates['collab_score'].min(), candidates['collab_score'].max()
    candidates['collab_norm']   = (candidates['collab_score'] - mn) / (mx - mn)
    candidates['hybrid_score']  = 0.4 * candidates['content_score'] + 0.6 * candidates['collab_norm']
    return candidates.sort_values('hybrid_score', ascending=False).head(num)

# ── CSS ──
st.markdown("""
<style>
.movie-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 10px;
    text-align: center;
    height: 100%;
    border: 1px solid #333;
}
.movie-card img {
    width: 100%;
    border-radius: 8px;
    height: 280px;
    object-fit: cover;
}
.movie-title {
    color: white;
    font-weight: bold;
    font-size: 13px;
    margin-top: 8px;
}
.movie-meta {
    color: #aaa;
    font-size: 11px;
}
.score-badge {
    background: #e50914;
    color: white;
    border-radius: 20px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ── UI ──
st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("Personalised recommendations based on **genre similarity** and **user taste** — with real movie posters!")

with st.spinner("Loading model... ⏳"):
    movies, ratings, cosine_sim, indices, svd = load_and_train()

st.success("✅ Model ready!")

# ── Tabs ──
tab1, tab2 = st.tabs(["🎯 Get Recommendations", "🔥 Popular Movies"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        movie_list = sorted(movies['title'].tolist())
        selected_movie = st.selectbox("🎥 Select a movie you like:", movie_list)
    with col2:
        user_id = st.number_input("👤 Enter your User ID (1–610):", min_value=1, max_value=610, value=1)

    num_recs = st.slider("📋 Number of recommendations:", 5, 20, 10)

    if st.button("🚀 Get Recommendations", use_container_width=True):
        with st.spinner("Finding best movies for you..."):
            results = hybrid_recommend(user_id, selected_movie, movies, cosine_sim, indices, svd, num=num_recs)

        if results is None:
            st.error("Movie not found.")
        else:
            st.markdown(f"### 🎯 Top {num_recs} recommendations for **{selected_movie}**")
            cols = st.columns(5)
            for i, row in enumerate(results.itertuples()):
                poster, overview, vote, year = fetch_poster(row.title)
                with cols[i % 5]:
                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/200x300?text=No+Poster", use_container_width=True)
                    st.markdown(f"**{row.title}**")
                    st.caption(f"⭐ {vote} | 📅 {year}")
                    st.caption(f"🎭 {row.genres[:30]}...")
                    st.caption(f"Score: {row.hybrid_score:.3f}")

with tab2:
    st.markdown("### 🔥 Popular Movies Right Now")
    with st.spinner("Fetching latest movies from TMDB..."):
        popular = fetch_popular_movies()

    cols = st.columns(5)
    for i, movie in enumerate(popular[:20]):
        with cols[i % 5]:
            poster_path = movie.get('poster_path')
            if poster_path:
                st.image(f"https://image.tmdb.org/t/p/w500{poster_path}", use_container_width=True)
            st.markdown(f"**{movie['title']}**")
            st.caption(f"⭐ {movie.get('vote_average', 0)} | 📅 {movie.get('release_date', '')[:4]}")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# ── Page config ──
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── Load & train (cached so it runs only once) ──
@st.cache_resource
def load_and_train():
    movies  = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')

    # Content-based
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    tfidf        = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
    cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices      = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    # Collaborative
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

# ── UI ──────────────────────────────────────────
st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("Get personalised movie recommendations based on **genre similarity** and **user taste**.")

with st.spinner("Loading model... please wait ⏳"):
    movies, ratings, cosine_sim, indices, svd = load_and_train()

st.success("✅ Model ready!")

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
        st.error("Movie not found. Please try another.")
    else:
        st.markdown(f"### 🎯 Top {num_recs} recommendations for **{selected_movie}**")
        for i, row in enumerate(results.itertuples(), 1):
            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{i}. {row.title}**")
                    st.caption(f"🎭 {row.genres}")
                with c2:
                    st.metric("Hybrid Score", f"{row.hybrid_score:.3f}")
                st.divider()
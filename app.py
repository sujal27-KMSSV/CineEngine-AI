import streamlit as st
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. PAGE CONFIG & STYLES
# ==========================================
st.set_page_config(
    page_title="CineEngine AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme aesthetic styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { color: #ffffff; }
    .movie-card {
        background-color: #1a1f2c;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #2e364f;
        margin-bottom: 15px;
    }
    .badge {
        background-color: #e50914;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .provider-badge {
        background-color: #2b3040;
        color: #61dafb;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-right: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "ratings" not in st.session_state:
    # Format: {movie_id: {"title": str, "rating": int, "genres": list}}
    st.session_state.ratings = {}
if "region" not in st.session_state:
    st.session_state.region = "US"
if "user_subscriptions" not in st.session_state:
    st.session_state.user_subscriptions = ["Netflix", "Amazon Prime Video", "Disney Plus"]

# TMDB API Setup (Checks Streamlit Secrets or Environment Variable)
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", os.getenv("TMDB_API_KEY", ""))

# ==========================================
# 3. FALLBACK DATASET (When No API Key Provided)
# ==========================================
MOCK_MOVIES = [
    {
        "id": 27205,
        "title": "Inception",
        "overview": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
        "genres": ["Action", "Science Fiction", "Adventure"],
        "vote_average": 8.4,
        "release_date": "2010-07-15",
        "poster_path": "https://image.tmdb.org/t/p/w500/oYuLE3932S32230C99A99BA5f2C.jpg",
        "providers": ["Netflix", "Max", "Apple TV (Rent)"]
    },
    {
        "id": 157336,
        "title": "Interstellar",
        "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
        "genres": ["Adventure", "Drama", "Science Fiction"],
        "vote_average": 8.4,
        "release_date": "2014-11-05",
        "poster_path": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
        "providers": ["Paramount Plus", "Amazon Prime Video"]
    },
    {
        "id": 155,
        "title": "The Dark Knight",
        "overview": "Batman raises the stakes in his war on crime with the help of Lt. Jim Gordon and District Attorney Harvey Dent.",
        "genres": ["Drama", "Action", "Crime"],
        "vote_average": 8.5,
        "release_date": "2008-07-16",
        "poster_path": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7v2aTXTo6.jpg",
        "providers": ["Max", "Apple TV (Rent)"]
    },
    {
        "id": 597,
        "title": "Titanic",
        "overview": "101-year-old Rose DeWitt Bukater tells the story of her lifetime aboard the Titanic.",
        "genres": ["Drama", "Romance"],
        "vote_average": 7.9,
        "release_date": "1997-11-18",
        "poster_path": "https://image.tmdb.org/t/p/w500/9xh8C3I0R4O8O942oR4Pz96xR.jpg",
        "providers": ["Paramount Plus", "Disney Plus"]
    },
    {
        "id": 19995,
        "title": "Avatar",
        "overview": "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting his home.",
        "genres": ["Action", "Adventure", "Fantasy", "Science Fiction"],
        "vote_average": 7.5,
        "release_date": "2009-12-15",
        "poster_path": "https://image.tmdb.org/t/p/w500/kyeqWdyUXW608A326E17i925H.jpg",
        "providers": ["Disney Plus"]
    }
]

# ==========================================
# 4. API & DATA FETCHING FUNCTIONS
# ==========================================
def fetch_tmdb_data(endpoint, params={}):
    """Generic fetch function for TMDB API v3."""
    if not TMDB_API_KEY:
        return None
    base_url = f"https://api.themoviedb.org/3/{endpoint}"
    default_params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    default_params.update(params)
    try:
        res = requests.get(base_url, params=default_params, timeout=5)
        return res.json() if res.status_code == 200 else None
    except Exception:
        return None

def get_trending_movies():
    """Fetch trending movies or fallback to mock list."""
    data = fetch_tmdb_data("trending/movie/day")
    if data and "results" in data:
        movies = []
        for item in data["results"][:12]:
            movies.append({
                "id": item["id"],
                "title": item.get("title", "Untitled"),
                "overview": item.get("overview", "No description available."),
                "genres": ["Action", "Drama", "Sci-Fi"], # Simplified for API view
                "vote_average": round(item.get("vote_average", 0), 1),
                "release_date": item.get("release_date", "N/A"),
                "poster_path": f"https://image.tmdb.org/t/p/w500{item['poster_path']}" if item.get("poster_path") else "https://via.placeholder.com/500x750?text=No+Poster"
            })
        return movies
    return MOCK_MOVIES

def get_watch_providers(movie_id, region="US"):
    """Fetch streaming availability via TMDB Watch Providers API."""
    data = fetch_tmdb_data(f"movie/{movie_id}/watch/providers")
    if data and "results" in data and region in data["results"]:
        reg_data = data["results"][region]
        providers = []
        if "flatrate" in reg_data:
            providers.extend([p["provider_name"] for p in reg_data["flatrate"]])
        if "rent" in reg_data:
            providers.extend([f"{p['provider_name']} (Rent)" for p in reg_data["rent"]])
        return list(set(providers))
    return ["Netflix", "Amazon Prime Video"] if not TMDB_API_KEY else ["Unavailable in Selected Region"]

# ==========================================
# 5. MACHINE LEARNING ENGINE (TF-IDF + COSINE SIMILARITY)
# ==========================================
def build_recommendation_engine(movies_list):
    """Builds TF-IDF feature matrix and Cosine Similarity matrix."""
    df = pd.DataFrame(movies_list)
    df["features"] = df["overview"] + " " + df["genres"].apply(lambda g: " ".join(g) if isinstance(g, list) else str(g))
    
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["features"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, similarity_matrix

def get_recommendations(movie_id, movies_list, top_n=3):
    """Retrieves top N similar movies using Cosine Similarity."""
    try:
        df, sim_matrix = build_recommendation_engine(movies_list)
        if movie_id not in df["id"].values:
            return movies_list[:top_n]
        
        idx = df[df["id"] == movie_id].index[0]
        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        recommended_indices = [i[0] for i in sim_scores[1:top_n+1]]
        return df.iloc[recommended_indices].to_dict("records")
    except Exception:
        return movies_list[:top_n]

# ==========================================
# 6. UI RENDER HELPER FUNCTIONS
# ==========================================
def render_movie_card(movie):
    """Standardized UI component for movie display."""
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.image(movie["poster_path"], use_container_width=True)
    with col2:
        st.subheader(movie["title"])
        st.caption(f"⭐ **{movie['vote_average']}/10** | 📅 {movie['release_date']}")
        
        # Display Genres
        genres = movie.get("genres", [])
        if genres:
            st.write(" **Genres:** " + ", ".join(genres) if isinstance(genres, list) else str(genres))
            
        st.write(movie["overview"][:140] + "...")
        
        # Display Streaming Availability
        providers = movie.get("providers", get_watch_providers(movie["id"], st.session_state.region))
        st.markdown("**Where to Watch:**")
        prov_html = "".join([f'<span class="provider-badge">{p}</span>' for p in providers[:3]])
        st.markdown(prov_html, unsafe_allow_html=True)
        
        st.write("")
        # Interactive Actions
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            in_watchlist = any(m["id"] == movie["id"] for m in st.session_state.watchlist)
            if st.button("➕ Watchlist" if not in_watchlist else "✔ In Watchlist", key=f"wl_{movie['id']}"):
                if not in_watchlist:
                    st.session_state.watchlist.append(movie)
                    st.toast(f"Added '{movie['title']}' to Watchlist!", icon="🎬")
                else:
                    st.session_state.watchlist = [m for m in st.session_state.watchlist if m["id"] != movie["id"]]
                    st.toast(f"Removed '{movie['title']}' from Watchlist.", icon="🗑️")
                st.rerun()
                
        with btn_col2:
            current_rating = st.session_state.ratings.get(movie["id"], {}).get("rating", 0)
            new_rating = st.selectbox(
                "Your Rating:",
                options=[0, 1, 2, 3, 4, 5],
                index=current_rating,
                key=f"rate_{movie['id']}"
            )
            if new_rating != current_rating:
                if new_rating > 0:
                    st.session_state.ratings[movie["id"]] = {
                        "title": movie["title"],
                        "rating": new_rating,
                        "genres": movie.get("genres", [])
                    }
                    st.toast(f"Rated '{movie['title']}' {new_rating}/5 Stars!", icon="⭐")
                elif movie["id"] in st.session_state.ratings:
                    del st.session_state.ratings[movie["id"]]
                st.rerun()

# ==========================================
# 7. MAIN APPLICATION NAVIGATION & LAYOUT
# ==========================================
st.sidebar.title("🎬 CineEngine AI")
menu = st.sidebar.radio("Navigation", ["Explore & Search", "AI Recommendations", "User Profile"])

# Sidebar Region & Subscription Filter
st.sidebar.divider()
st.sidebar.subheader("🌍 Regional Settings")
st.session_state.region = st.sidebar.selectbox("Select Watch Region", ["US", "UK", "IN", "CA", "AU"], index=0)

if not TMDB_API_KEY:
    st.sidebar.warning("⚠️ Running on Offline Fallback Dataset. Set TMDB_API_KEY in Secrets for live data.")

movies_dataset = get_trending_movies()

# ------------------------------------------
# PAGE 1: EXPLORE & SEARCH
# ------------------------------------------
if menu == "Explore & Search":
    st.title("🍿 Discover Movies")
    
    # Search Bar
    search_query = st.text_input("🔍 Search movies by title or keyword...", "")
    
    if search_query:
        # Search via TMDB API or Fallback Filter
        if TMDB_API_KEY:
            results = fetch_tmdb_data("search/movie", {"query": search_query})
            if results and "results" in results:
                movies_dataset = [{
                    "id": m["id"],
                    "title": m.get("title", ""),
                    "overview": m.get("overview", "No details available."),
                    "genres": ["Movie"],
                    "vote_average": round(m.get("vote_average", 0), 1),
                    "release_date": m.get("release_date", "N/A"),
                    "poster_path": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else "https://via.placeholder.com/500x750"
                } for m in results["results"][:10]]
        else:
            movies_dataset = [m for m in MOCK_MOVIES if search_query.lower() in m["title"].lower() or search_query.lower() in m["overview"].lower()]

    st.subheader("Trending Today" if not search_query else f"Search Results for '{search_query}'")
    
    # Render 2-Column Grid
    col_a, col_b = st.columns(2)
    for idx, movie in enumerate(movies_dataset):
        with col_a if idx % 2 == 0 else col_b:
            render_movie_card(movie)

# ------------------------------------------
# PAGE 2: AI RECOMMENDATIONS
# ------------------------------------------
elif menu == "AI Recommendations":
    st.title("🤖 AI Recommendation Engine")
    st.caption("Powered by Content-Based Filtering & Cosine Similarity Matrix")
    
    selected_movie_name = st.selectbox("Select a movie you love:", [m["title"] for m in movies_dataset])
    target_movie = next(m for m in movies_dataset if m["title"] == selected_movie_name)
    
    st.subheader(f"Because you liked '{selected_movie_name}':")
    recs = get_recommendations(target_movie["id"], movies_dataset, top_n=3)
    
    cols = st.columns(3)
    for idx, movie in enumerate(recs):
        with cols[idx % 3]:
            st.image(movie["poster_path"], use_container_width=True)
            st.markdown(f"**{movie['title']}**")
            st.caption(f"⭐ {movie['vote_average']}/10")
            st.write(movie["overview"][:90] + "...")

# ------------------------------------------
# PAGE 3: USER PROFILE DASHBOARD
# ------------------------------------------
elif menu == "User Profile":
    st.title("👤 User Profile Dashboard")
    
    # User Stats Summary Cards
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("Movies Rated", len(st.session_state.ratings))
    stat_col2.metric("Watchlist Total", len(st.session_state.watchlist))
    stat_col3.metric("Selected Region", st.session_state.region)
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📌 Watchlist & Favorites", "⭐ My Ratings", "⚙️ Streaming Subscriptions"])
    
    with tab1:
        if not st.session_state.watchlist:
            st.info("Your Watchlist is empty. Browse 'Explore & Search' to add movies!")
        else:
            for item in st.session_state.watchlist:
                st.markdown(f"**{item['title']}** (⭐ {item['vote_average']}/10)")
                
    with tab2:
        if not st.session_state.ratings:
            st.info("You haven't rated any movies yet!")
        else:
            for m_id, data in st.session_state.ratings.items():
                st.write(f"🍿 **{data['title']}** — Given Score: **{data['rating']}/5 Stars**")
                
    with tab3:
        st.subheader("Manage Active Subscriptions")
        subs = st.multiselect(
            "Filter availability based on your active services:",
            ["Netflix", "Amazon Prime Video", "Disney Plus", "Max", "Hulu", "Paramount Plus", "Apple TV"],
            default=st.session_state.user_subscriptions
        )
        st.session_state.user_subscriptions = subs
        st.success("Streaming preferences saved successfully!")

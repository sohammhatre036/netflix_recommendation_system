import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Team Code Hulk

# Load Netflix logo
netflix_logo = "https://upload.wikimedia.org/wikipedia/commons/7/75/Netflix_icon.svg"

# Load dataset
url = "https://raw.githubusercontent.com/sohammhatre036/netflix_recommendation_system/main/netflix_titles_cleaned.csv"
df = pd.read_csv(url)

# Fill missing values
df.fillna("", inplace=True)

# Combine relevant features into a single string
df["combined_features"] = df["cast"] + " " + df["listed_in"] + " " + df["description"] 

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_features"])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, content_type="All", num_recommendations=5):
    title = title.strip().lower()
    
    # Find closest match
    matching_titles = df[df["title"].str.lower().str.contains(title, na=False)]
    
    if matching_titles.empty:
        return ["❌ Movie not found! Try another title."]
    
    idx = matching_titles.index[0]

    # Get similarity scores for all items
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort items by similarity score (excluding the movie itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    # Get recommended movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Filter by type (Movie / TV Show)
    if content_type != "All":
        movie_indices = [i for i in movie_indices if df.iloc[i]["type"] == content_type]

    recommendations = df.iloc[movie_indices][["title", "country", "description"]]

    return recommendations if not recommendations.empty else ["⚠️ No similar movies found."]

# Streamlit UI
st.set_page_config(page_title="Netflix Recommender - Code Hulk", page_icon=netflix_logo, layout="wide")

# Netflix Logo
st.image(netflix_logo, width=80)
st.title("🎬 Netflix Movie Recommendation System")
st.markdown("### By **Code Hulk**")
st.write("Start typing a movie name to get  recommendations!")

# Movie Title Autocomplete
movie_list = df["title"].tolist()
selected_movie = st.selectbox("Enter or select a movie:", [""] + movie_list)

# Dropdown filter (Movie / TV Show)
# content_type = st.selectbox("Filter by type:", ["All", "Movie", "TV Show"])

if st.button("🔍 Get Recommendations"):
    if selected_movie:
        recommendations = get_recommendations(selected_movie, num_recommendations=5)

        if isinstance(recommendations, list):
            st.error(recommendations[0])  # Display error message
        else:
            st.subheader("🎥 **Recommended Titles:**")
            for _, row in recommendations.iterrows():
                st.markdown(f"**🎬 {row['title']}** ({row['country']})")
                st.write(f"📜 {row['description'][:200]}...")  # Show first 200 characters
                st.write("---")
    else:
        st.warning("⚠️ Please select or type a movie title.")

# Footer
st.markdown("---")


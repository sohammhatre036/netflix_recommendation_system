import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
url = "https://raw.githubusercontent.com/sohammhatre036/netflix_recommendation_system/main/netflix_titles_cleaned.csv"
df = pd.read_csv(url)


# Fill missing values
df["cast"] = df["cast"].fillna("")
df["listed_in"] = df["listed_in"].fillna("")
df["description"] = df["description"].fillna("")

# Combine relevant features into a single string
df["combined_features"] = df["cast"] + " " + df["listed_in"] + " " + df["description"] + " " + df["country"]

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_features"])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, num_recommendations=5):
    title = title.strip().lower()
    
    if title not in df["title"].str.lower().values:
        return ["❌ Movie not found! Try another title."]
    
    idx = df[df["title"].str.lower() == title].index[0]
    
    # Get similarity scores for all items
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort items by similarity score (excluding the movie itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    # Get recommended movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    return df["title"].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie title to get recommendations!")

# User input for movie title
user_input = st.text_input("Enter Movie Title:", "")

if st.button("Get Recommendations"):
    if user_input:
        recommendations = get_recommendations(user_input, num_recommendations=5)
        st.write("🎥 **Recommended Movies:**")
        for movie in recommendations:
            st.write(f"- {movie}")
    else:
        st.warning("⚠️ Please enter a movie title.")

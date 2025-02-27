import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
url = "https://raw.githubusercontent.com/sohammhatre036/netflix_recommendation_system/main/netflix_titles_cleaned.csv"
df = pd.read_csv(url)

# Fill missing values
df.fillna("", inplace=True)

# Combine relevant features into a single string
df["combined_features"] = df["listed_in"] + " " + df["cast"] + " " + df["description"]

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_features"].fillna(""))

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Streamlit UI
st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Netflix Movie & TV Show Recommender")
st.markdown("### **Find what to watch next!**")

# User selects whether they want Movie or TV Show recommendations
content_type = st.radio("Do you want recommendations for a **Movie** or a **TV Show**?", ("Movie", "TV Show"))

# Movie Title Autocomplete
movie_list = df["title"].tolist()
selected_movie = st.selectbox("Enter or select a Movie/TV Show:", [""] + movie_list)

# Function to get recommendations
def get_recommendations(title, content_type, df, cosine_sim, num_recommendations=10):
    matching_titles = df[df["title"].str.lower() == title.lower()]
    
    if matching_titles.empty:
        return ["❌ Movie/TV Show not found! Try another title."]

    idx = matching_titles.index[0]
    
    # Get similarity scores for all titles
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]  # Exclude itself

    # Get recommended movie/show indices
    movie_indices = [i[0] for i in sim_scores]

    # Filter by chosen type (Movie or TV Show)
    filtered_recommendations = df.iloc[movie_indices]
    filtered_recommendations = filtered_recommendations[filtered_recommendations["type"] == content_type]

    # Ensure genre relevance
    selected_genres = set(df.loc[idx, "listed_in"].split(", "))
    
    def genre_match(genres):
        return any(genre in selected_genres for genre in genres.split(", "))

    filtered_recommendations = filtered_recommendations[filtered_recommendations["listed_in"].apply(genre_match)]

    return filtered_recommendations.head(num_recommendations)[["title", "country", "listed_in", "description", "cast"]]

# Show recommendations when the user clicks the button
if st.button("🔍 Get Recommendations"):
    if selected_movie:
        recommendations = get_recommendations(selected_movie, content_type, df, cosine_sim)

        if isinstance(recommendations, list):
            st.error(recommendations[0])  # Display error message
        else:
            st.subheader("🎥 **Recommended Titles:**")
            for _, row in recommendations.iterrows():
                st.markdown(f"**🎬 {row['title']}** ({row['country']})")
                st.write(f"📜 {row['description'][:500]}...")  # Show first 300 characters
                st.write(f"🎭 **Genres:** {row['listed_in']}")
                st.write(f"🎭 **Cast:** {row['cast'][:300]}...")  # Show first 300 characters
                st.write("---")
    else:
        st.warning("⚠️ Please select or type a Movie/TV Show title.")

# Footer
st.markdown("---")
st.markdown("🎬 **Netflix Recommender by Code Hulk**")

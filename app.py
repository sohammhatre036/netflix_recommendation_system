import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (Ensure you have a CSV file with the necessary columns)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/sohammhatre036/netflix_recommendation_system/main/netflix_titles_cleaned.csv"
    df = pd.read_csv(url)
    df.fillna("", inplace=True)
    return df

df = load_data()

# Combine relevant features into a single string
df['combined_features'] = df['cast'] + " " + df['listed_in'] + " " + df['description'] + " " + df['country']

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_items = sim_scores[1:11]
    return df['title'].iloc[[i[0] for i in top_items]]

# Streamlit UI
st.title("Movie Recommendation System")

selected_movie = st.selectbox("Choose a movie/show", df['title'].unique())

if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_movie)
    if recommendations:
        st.write("### Recommended Titles:")
        for movie in recommendations:
            st.write(f"- {movie}")
    else:
        st.write("No recommendations found.")

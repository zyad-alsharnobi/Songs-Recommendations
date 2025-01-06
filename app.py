import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load the model and scaler separately
@st.cache_resource
def load_model():
    with open('mood_classifier_model.pkl', 'rb') as file:
        clf = pickle.load(file)
    with open('mood_classifier_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return clf, scaler

clf, scaler = load_model()

# Load the dataset (for recommendations)
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv('preprocessed.csv')

data = load_data()
data = data.drop_duplicates(subset=['name', 'artists'])

# Debugging: Check if 'spotify_id' column exists
if 'spotify_id' not in data.columns:
    st.error("Error: The dataset is missing the 'spotify_id' column. Please ensure your dataset includes Spotify track IDs.")
    st.stop()

# Normalize the features
scaler = MinMaxScaler()
data[['danceability', 'energy', 'valence', 'acousticness', 'tempo']] = scaler.fit_transform(
    data[['danceability', 'energy', 'valence', 'acousticness', 'tempo']]
)

# Define mood mapping (mood to feature ranges)
mood_ranges = {
    'calm': {
        'danceability': (0.2, 0.5),
        'energy': (0.1, 0.4),
        'valence': (0.3, 0.6),
        'acousticness': (0.7, 1.0),
        'tempo': (0.3, 0.6)
    },
    'energetic': {
        'danceability': (0.7, 1.0),
        'energy': (0.8, 1.0),
        'valence': (0.6, 1.0),
        'acousticness': (0.0, 0.3),
        'tempo': (0.7, 1.0)
    },
    'happy': {
        'danceability': (0.6, 1.0),
        'energy': (0.6, 1.0),
        'valence': (0.7, 1.0),
        'acousticness': (0.1, 0.4),
        'tempo': (0.6, 1.0)
    },
    'sad': {
        'danceability': (0.1, 0.4),
        'energy': (0.1, 0.3),
        'valence': (0.1, 0.4),
        'acousticness': (0.6, 1.0),
        'tempo': (0.2, 0.5)
    },
    'love/romance': {
        'danceability': (0.5, 0.8),
        'energy': (0.4, 0.7),
        'valence': (0.6, 1.0),
        'acousticness': (0.2, 0.5),
        'tempo': (0.5, 0.8)
    },
    'study': {
        'danceability': (0.1, 0.4),
        'energy': (0.1, 0.4),
        'valence': (0.3, 0.6),
        'acousticness': (0.7, 1.0),
        'tempo': (0.3, 0.6)
    },
    'party': {
        'danceability': (0.8, 1.0),
        'energy': (0.8, 1.0),
        'valence': (0.7, 1.0),
        'acousticness': (0.0, 0.3),
        'tempo': (0.8, 1.0)
    },
    'relaxation': {
        'danceability': (0.1, 0.4),
        'energy': (0.1, 0.3),
        'valence': (0.2, 0.5),
        'acousticness': (0.8, 1.0),
        'tempo': (0.2, 0.5)
    }
}

# Function to generate random mood features
def generate_mood_features(mood):
    features = {}
    for feature, (min_val, max_val) in mood_ranges[mood].items():
        features[feature] = np.random.uniform(min_val, max_val)
    return features


# Function to generate random mood features
def generate_mood_features(mood):
    features = {}
    for feature, (min_val, max_val) in mood_ranges[mood].items():
        features[feature] = np.random.uniform(min_val, max_val)
    return features

# Define region mapping (region to country codes)
region_mapping = {
    
    'English (US)': ['US'],  # United States
    'English (CA)': ['CA'],  # Canada
    'English (GB)': ['GB'],  # United Kingdom
    'Arabic': ['EG'], # Arab
}

# Streamlit app
st.title("🎵 Mood-Based Song Recommender")

# User input for region
region = st.selectbox("Select your region:", list(region_mapping.keys()))

# Filter the dataset based on the selected region
country_codes = region_mapping[region]
filtered_data = data[data['country'].isin(country_codes)]

# Debugging: Check if 'spotify_id' column exists in filtered data
if 'spotify_id' not in filtered_data.columns:
    st.error("Error: The filtered dataset is missing the 'spotify_id' column. Please check your dataset and filtering logic.")
    st.stop()

# User input for mood
mood = st.selectbox("Select your mood:", ['calm', 'energetic', 'happy', 'sad', 'love/romance', 'study', 'party', 'relaxation'])

# Initialize session state for mood features and recommendations
if 'mood_features' not in st.session_state:
    st.session_state.mood_features = generate_mood_features(mood)
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Optimized function to recommend songs
def recommend_songs(data, mood_features, num_recommendations=5):
    # Create a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()
    
    # Convert mood_features to a numpy array
    mood_vector = np.array(list(mood_features.values()))
    
    # Extract the relevant features from the dataset
    features = data_copy[['danceability', 'energy', 'valence', 'acousticness', 'tempo']].values
    
    # Compute the Euclidean distance using vectorized operations
    distances = np.sqrt(np.sum((features - mood_vector) ** 2, axis=1))
    
    # Add the distances to the copied DataFrame
    data_copy['similarity'] = distances
    
    # Sort the songs by similarity and return the top recommendations
    recommendations = data_copy.sort_values(by='similarity').head(num_recommendations)
    
    # Debugging: Print the top recommendations and their similarity scores
    print(f"Top recommendations for {mood} in {region}:")
    print(recommendations[['name', 'artists', 'similarity', 'spotify_id']])
    
    return recommendations[['name', 'artists', 'popularity', 'spotify_id']]

# Button to get recommendations
if st.button("Recommend Songs"):
    st.session_state.mood_features = generate_mood_features(mood)
    st.session_state.recommendations = recommend_songs(filtered_data, st.session_state.mood_features)

# Display recommendations if they exist
if st.session_state.recommendations is not None:

    # Display Spotify embed player for each recommended song
    st.write("### Play the Songs")
    for _, row in st.session_state.recommendations.iterrows():
        spotify_id = row['spotify_id']
        embed_url = f"https://open.spotify.com/embed/track/{spotify_id}"
        st.components.v1.html(
            f'<iframe src="{embed_url}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>',
            height=100,
        )

    # Refresh button to get new recommendations
    if st.button("Refresh Recommendations"):
        st.session_state.mood_features = generate_mood_features(mood)
        st.session_state.recommendations = recommend_songs(filtered_data, st.session_state.mood_features)
        st.rerun()  # Refresh the app to display new recommendations

# Optional: Add some styling or descriptions
st.markdown("""
    ### How it works:
    1. Select your region and mood from the dropdowns.
    2. Click the "Recommend Songs" button.
    3. If you don't like the recommendations, click "Refresh Recommendations" to get a new set.
    4. Enjoy your personalized song recommendations!
""")
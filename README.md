# Moodify :headphones:
[Live Demo]() | [Paper](https://github.com/digitalnoma/moodify-recommender/blob/main/assets/50_038_CDS_Group_08.pdf) | [Main Assets](https://www.dropbox.com/scl/fo/gdj3c7clw9egnjyt9iu36/AN_fm12u-WLkyOGWqWRm7Xo?rlkey=7v28k1ef6fvw0icu6t1070rb4&dl=0)

## Overview
Moodify is a web-based application designed to analyze audio files to detect the mood conveyed by the music and predict its genre. Utilizing advanced machine learning models, Moodify extracts audio features and classifies them into predefined categories of emotions and musical genres, providing a user-friendly interface for uploading audio, visualizing emotion through a valence-arousal plot, and identifying the music genre.

## Quickstart Guide: Local Deployment

### Step 1: Clone the Repository
To get started with Moodify locally, clone the repository to your machine using the following Git command:
```
git clone https://github.com/your-repository/moodify.git
cd moodify
```


### Step 2: Set Up Virtual Environment
Create a virtual environment to manage the project's dependencies separately from your main Python installation:
```
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate
```

### Step 3: Install Dependencies
Install all required dependencies by running:
```
pip install -r requirements.txt
```

### Step 4: Run the Application
Finally, launch the application using Streamlit:
```
streamlit run runner.py
```

## Features

### Audio Upload and Playback
- **Upload:** Users can upload audio files in MP3 format.
- **Playback:** The app provides an audio player for users to listen to their uploaded tracks.

### Emotion Detection
- **Feature Extraction:** Utilizes librosa to perform advanced audio feature extraction.
- **Mood Prediction:** Based on the extracted features, the app predicts valence and arousal, categorizing the mood of the music.

### Genre Prediction
- **Genre Analysis:** Analyzes the audio features to predict the music's genre using a pre-trained neural network model.

### Visual Display
- **Valence-Arousal Plot:** Displays the valence and arousal values on a graph, allowing users to visualize the emotional content of their music.
- **Shape Selection:** Users can select different shapes (circle, line, triangle, parabola) which affects how mood data is projected and visualized.

### Interactive Shape Selection
After the music has been processed, users are prompted to select a geometric shape to view the valence and arousal data projected differently, enhancing the interaction and visualization experience.

## Conclusion
Moodify leverages cutting-edge technology to offer users a comprehensive tool for exploring the emotional and genre-based characteristics of music. This application serves as a powerful platform for music analysis, providing insights into the emotional depths of audio tracks through an interactive and engaging interface.

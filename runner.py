import librosa
import numpy as np
import pandas as pd
import math
import altair as alt
import time
import os
import torch
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapes import MusicOnTrajectory, Line, Circle, Triangle, Parabola
from audio_models import Attention, AudioNet, Predictor, MusicGenreClassifier
from constants import emotions, clustered_emotions

# LOAD VA GENERATION
def extract_features(audio_path, sample_rate=44100):
    wave, sr = librosa.load(audio_path, sr=sample_rate)
    if len(wave) < sr * 45:
        wave = np.pad(wave, (0, sr * 45 - len(wave)), 'constant')
    wave = wave[:sr * 45]

    hop_length = int(sr * 0.01)
    win_length = int(sr * 0.025)

    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=20, n_fft=2048, hop_length=hop_length, win_length=win_length)
    chroma = librosa.feature.chroma_stft(y=wave, sr=sr, n_fft=2048, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=wave, sr=sr, n_fft=2048, hop_length=hop_length)
    return np.concatenate((mfcc, chroma, contrast), axis=0)

def predict(model, features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(features)
    return output.cpu().numpy()

def find_emotion(valence, arousal):
    closest_emotion = None
    min_distance = math.inf

    for emotion, scores in emotions.items():
        distance = math.sqrt((valence - scores["valence"])**2 + (arousal - scores["arousal"])**2)

        if distance < min_distance:
            min_distance = distance
            closest_emotion = emotion

    return closest_emotion

def get_colormap(valence, arousal):
    valence, arousal = normalize_value(valence), normalize_value(arousal)
    emotion = find_emotion(valence, arousal)
    for color, emotion_list in clustered_emotions.items():
        if emotion in emotion_list:
            return color
    return None

def normalize_value(value):
    return (value - 1) / 4 - 1


# LOAD GENRE PREDICTION
model = MusicGenreClassifier(input_size=57, num_classes=10)
model.load_state_dict(torch.load('./model_checkpoints/genre_classifier_model.pth'))
model.eval()

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = []
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend([np.mean(chroma_stft), np.var(chroma_stft)])
    rms = librosa.feature.rms(y=y)
    features.extend([np.mean(rms), np.var(rms)])
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([np.mean(spec_centroid), np.var(spec_centroid)])
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.extend([np.mean(spec_bandwidth), np.var(spec_bandwidth)])
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend([np.mean(rolloff), np.var(rolloff)])
    zero_cross_rate = librosa.feature.zero_crossing_rate(y)
    features.extend([np.mean(zero_cross_rate), np.var(zero_cross_rate)])
    harmony = librosa.effects.harmonic(y)
    features.extend([np.mean(harmony), np.var(harmony)])
    percussive = librosa.effects.percussive(y)
    features.extend([np.mean(percussive), np.var(percussive)])
    tempo = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
    features.append(np.mean(tempo))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for mfcc in mfccs:
        features.extend([np.mean(mfcc), np.var(mfcc)])

    return np.array(features)

# LOAD DATASET
spotify_va = pd.read_csv("spotify_va.csv")

import base64
import os

# Function to convert image to base64
def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# List image shape paths
image_paths = ["./assets/circle.png", "./assets/line.png", "./assets/parabola.png", "./assets/triangle.png"]

# Verify images exist and convert to base64
images_base64 = []
for img_path in image_paths:
    if os.path.isfile(img_path):
        images_base64.append(get_image_as_base64(img_path))
    else:
        st.error(f"Image {img_path} not found in the directory.")
        st.stop()

# Define Valence Arousal Models
model_path_valence = './model_checkpoints/model_valence.pth'
model_path_arousal = './model_checkpoints/model_arousal.pth'

def filter_genre(df, genre):
  if genre == "blues" or genre == "jazz":
    filtered_df = df[df["genre"] == "blues"]
  elif genre == "raggae" or genre == "classical":
    filtered_df = df
  else:
    filtered_df = df[df["genre"] == genre]
  return filtered_df

# Function to display the shapes
from scroll_utils import inject_scroll_to_bottom

def display_images(genre, valence, arousal, color, spotify_va):
    print(f'displayed images!')
    # Each row has 2 columns, so we create two rows
    row1_cols = st.columns(2)
    row2_cols = st.columns(2)

    if 'image_clicked' not in st.session_state:
        st.session_state.image_clicked = [False, False, False, False]

    idx = 0
    for col in row1_cols + row2_cols:
        with col:
            # Create a new column layout for the title and the button
            title_col, button_col = st.columns([3, 1]) 

            with title_col:
                # Extract the shape name from the image filename
                shape = image_paths[idx].split('/')[-1].split('.')[0]
                # Display the shape name as a title for each image
                st.subheader(f"{shape.capitalize()} Projection")  # Use subheader for better visual separation

            with button_col:
                button_key = f'click_{idx + 1}'
                if st.button('Select', key=button_key):
                    st.session_state.image_clicked[idx] = True

            # Display the image below the title and button
            st.image(image_paths[idx], use_column_width=True)
            idx += 1

    # Check if any image was clicked
    for index, clicked in enumerate(st.session_state['image_clicked']):
        if clicked:
            # Reset the clicked states for all images
            st.session_state.image_clicked = [False] * len(st.session_state.image_clicked)
            # Clear the previous output (if any)
            if 'plot_placeholder' in st.session_state:
                st.session_state.plot_placeholder.empty()

            # Store the new plot placeholder
            st.session_state.plot_placeholder = st.empty()
            shape = image_paths[index].split('/')[-1].split('.')[0]

            new_row = {
                'spotify_id': 'new_id',
                'artist': 'New Artist',
                'track': None,
                'file_path': None,
                'genre': genre,
                'valence': valence,
                'arousal': arousal,
                'colour': color
            }

            # Add the new row to the DataFrame
            spotify_va = spotify_va._append(new_row, ignore_index=True)
            filtered_df = filter_genre(spotify_va, genre)
            point = (valence, arousal)

            try:
                if "circle" in shape:
                    c= Circle(point)
                    t1 = MusicOnTrajectory(filtered_df, c)


                elif "line" in shape:
                    l = Line(point) 
                    t1 = MusicOnTrajectory(filtered_df, l)

                elif "parabola" in shape:
                    p = Parabola(point)
                    t1 = MusicOnTrajectory(filtered_df, p)

                else:
                    t = Triangle(point) 
                    t1 = MusicOnTrajectory(filtered_df, t)

                fig = t1.run()
                print("run() method completed, plotly.go figure returned.")

                if fig:
                    st.subheader('Valence-Arousal Graph')
                    st.plotly_chart(fig)
                    inject_scroll_to_bottom() # Scroll to bottom after rendering
                else:
                    print('No figure was created.')

            except Exception as e:
                print(f"An error occurred: {e}")

def save_upload(uploadedFile):
    with open('audio_upload.mp3', 'wb') as f:
            f.write(uploadedFile.read())

    # Determine audio file path
    return os.path.abspath('audio_upload.mp3')

def predict_genre(model, features):
        feature_values = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            outputs = model(feature_values)
            predicted_genre_index = outputs.argmax(dim=1).item()

        genre_mapping = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
                        5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

        return genre_mapping[predicted_genre_index]

def main():
    st.markdown("<h2 style='text-align: center;'>🎧 Moodify</h2>", unsafe_allow_html=True)

    audio_file = st.file_uploader("Please upload an audio file (MP3)", type=["mp3"])

    # If an audio file is uploaded, display an audio player
    if audio_file is not None and 'audio_processed' not in st.session_state:
        # Save audio file into a temporary file
        temp_audio_path = save_upload(audio_file)
        
        # Display the audio player for the uploaded file
        audio_title = audio_file.name.split(".")[0]  
        st.subheader(f"Audio Track: {audio_title}")
        st.audio(temp_audio_path, format='audio/mp3')

        # Initialise Predictor Object
        predictor = Predictor(model_path_valence, model_path_arousal)

        # Progress bar
        progress_text = "Emotion detection and genre prediction in progress. Please wait a couple of seconds."
        my_bar = st.progress(0, text=progress_text)

        # Emulating progress
        for percent_complete in range(100):
            time.sleep(0.1) 
            my_bar.progress(percent_complete + 1, text=progress_text)

        # Run emotion detection
        predictor = Predictor(model_path_valence, model_path_arousal)
        valence, arousal = predictor.predict(temp_audio_path)
        color = get_colormap(valence, arousal)

        print(f"Valence: {valence}, Arousal: {arousal}")
        print(f'Emotion Detected: {color}')

        features = extract_features(temp_audio_path)
        predicted_genre = predict_genre(model, features)
        
        # Store the results in session state
        st.session_state['audio_processed'] = True
        st.session_state['valence'] = valence
        st.session_state['arousal'] = arousal
        st.session_state['color'] = color
        st.session_state['genre'] = predicted_genre

        # Display the results
        # st.write(f"Emotion Detected: {color}")
        st.subheader(f"The predicted genre of the song is: {predicted_genre}")
        my_bar.empty()


    # If a file has been processed, offer shape selection
    if 'audio_processed' in st.session_state:
        st.subheader("Now, pick a shape!")
        st.write("Please scroll down to view your selection")
        display_images(st.session_state['genre'], st.session_state['valence'], st.session_state['arousal'], st.session_state['color'], spotify_va)

# Run runner.py
if __name__ == "__main__":
    main()
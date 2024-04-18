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

# LOAD VA GENERATION
class Attention(nn.Module): 
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        scores = self.attention(x)
        alpha = F.softmax(scores, dim=1)
        attended_features = x * alpha
        return attended_features.view(-1, self.feature_dim)

class AudioNet(nn.Module):
    def __init__(self, params_dict):
        super(AudioNet, self).__init__()
        self.in_ch = params_dict.get('in_ch', 1)
        self.num_filters1 = params_dict.get('num_filters1', 32)
        self.num_filters2 = params_dict.get('num_filters2', 64)
        self.num_hidden = params_dict.get('num_hidden', 128)
        self.out_size = params_dict.get('out_size', 1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.in_ch, self.num_filters1, kernel_size=10, stride=1),
            nn.BatchNorm1d(self.num_filters1),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.num_filters1, self.num_filters2, kernel_size=10, stride=1),
            nn.BatchNorm1d(self.num_filters2),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        self.pool = nn.AvgPool1d(kernel_size=10, stride=10)

        self._to_linear = None
        self.attention = Attention(self._get_to_linear())

        self.fc1 = nn.Linear(self._get_to_linear(), self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden, self.out_size)
        self.drop = nn.Dropout(p=0.5)
        self.act = nn.ReLU(inplace=True)

    def _get_to_linear(self):
        if self._to_linear is None:
            x = torch.randn(1, self.in_ch, 4501)
            with torch.no_grad():
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.pool(x)
                self._to_linear = x.numel() // x.shape[0]
        return self._to_linear

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, self._get_to_linear())
        x = self.attention(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.to(x.device)

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

class Predictor:
    def __init__(self, model_path_valence, model_path_arousal):
        self.model_path_valence = model_path_valence
        self.model_path_arousal = model_path_arousal
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        valence_params = {
            "in_ch": 39, "num_filters1": 32, "num_filters2": 64, "num_hidden": 64, "out_size": 1
        }
        arousal_params = {
            "in_ch": 39, "num_filters1": 32, "num_filters2": 32, "num_hidden": 128, "out_size": 1
        }

        self.valence_model = self.load_model(model_path_valence, valence_params)
        self.arousal_model = self.load_model(model_path_arousal, arousal_params)

    def load_model(self, model_path, params):
        model = AudioNet(params)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def extract_features(self, audio_path):
        sample_rate = 44100
        wave, sr = librosa.load(audio_path, sr=sample_rate)
        if len(wave) < sr * 45:
            wave = np.pad(wave, (0, sr * 45 - len(wave)), 'constant')
        wave = wave[:sr * 45]

        hop_length = int(sr * 0.01)
        win_length = int(sr * 0.025)

        mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=20, n_fft=2048, hop_length=hop_length, win_length=win_length)
        chroma = librosa.feature.chroma_stft(y=wave, sr=sr, n_fft=2048, hop_length=hop_length)
        contrast = librosa.feature.spectral_contrast(y=wave, sr=sr, n_fft=2048, hop_length=hop_length)

        features = np.concatenate((mfcc, chroma, contrast), axis=0)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return features_tensor.to(self.device)

    def predict(self, audio_path):
        features = self.extract_features(audio_path)
        with torch.no_grad():
            valence_prediction = self.valence_model(features)
            arousal_prediction = self.arousal_model(features)
        return valence_prediction.item(), arousal_prediction.item()

# EMOTION MAPPING
emotions = {
    "Sleepy": {"valence": 0.01, "arousal": -1.00},
    "Tired": {"valence": -0.01, "arousal": -1.00},
    "Afraid": {"valence": -0.12, "arousal": 0.79},
    "Angry":{"valence": -0.40, "arousal": 0.79},
    "Calm":{"valence": 0.78, "arousal": -0.68},
    "Relaxed":{"valence": 0.71, "arousal": -0.65},
    "Content":{"valence": 0.81, "arousal": -0.55},
    "Depressed":{"valence": -0.81, "arousal": -0.48},
    "Discontent":{"valence": -0.68, "arousal": -0.32},
    "Determined":{"valence": 0.73, "arousal": 0.26},
    "Happy":{"valence": 0.89, "arousal": 0.17},
    "Anxious":{"valence": -0.72, "arousal": -0.80},
    "Good":{"valence": 0.90, "arousal": -0.08},
    "Pensive":{"valence": 0.03, "arousal": -0.60},
    "Impressed":{"valence": 0.39, "arousal": -0.06},
    "Frustrated":{"valence": -0.60, "arousal": 0.40},
    "Disappointed":{"valence": -0.80, "arousal": -0.03},
    "Bored":{"valence": -0.35, "arousal": -0.78},
    "Annoyed":{"valence": -0.44, "arousal": 0.76},
    "Enraged":{"valence": -0.18, "arousal": 0.83},
    "Excited":{"valence": 0.70, "arousal": 0.71},
    "Melancholy":{"valence": -0.05, "arousal": -0.65},
    "Satisfied":{"valence": 0.77, "arousal": -0.63},
    "Distressed":{"valence": -0.71, "arousal": 0.55},
    "Uncomfortable":{"valence": -0.68, "arousal": -0.37},
    "Worried":{"valence": -0.07, "arousal": -0.32},
    "Amused":{"valence": 0.55, "arousal": 0.19},
    "Apathetic":{"valence": -0.20, "arousal": -0.12},
    "Peaceful":{"valence": 0.55, "arousal": -0.80},
    "Contemplative":{"valence": 0.58, "arousal": -0.60},
    "Embarrassed":{"valence": -0.31, "arousal": -0.60},
    "Sad":{"valence": -0.81, "arousal": -0.40},
    "Hopeful":{"valence": 0.61, "arousal": -0.30},
    "Pleased":{"valence": 0.89, "arousal": -0.10},
}

def find_emotion(valence, arousal):
    closest_emotion = None
    min_distance = math.inf

    for emotion, scores in emotions.items():
        distance = math.sqrt((valence - scores["valence"])**2 + (arousal - scores["arousal"])**2)

        if distance < min_distance:
            min_distance = distance
            closest_emotion = emotion

    return closest_emotion

clustered_emotions = {'blue': ['Determined',
  'Happy',
  'Good',
  'Impressed',
  'Excited',
  'Amused',
  'Hopeful',
  'Pleased'],
 'red': ['Depressed',
  'Discontent',
  'Anxious',
  'Disappointed',
  'Bored',
  'Uncomfortable',
  'Worried',
  'Apathetic',
  'Embarrassed',
  'Sad'],
 'green': ['Afraid', 'Angry', 'Frustrated', 'Annoyed', 'Enraged', 'Distressed'],
 'purple': ['Sleepy',
  'Tired',
  'Calm',
  'Relaxed',
  'Content',
  'Pensive',
  'Melancholy',
  'Satisfied',
  'Peaceful',
  'Contemplative']}

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
class MusicGenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicGenreClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

model = MusicGenreClassifier(input_size=57, num_classes=10)
model.load_state_dict(torch.load('genre_classifier_model.pth'))
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

# List your image paths
image_paths = ["./assets/circle.png", "./assets/line.png", "./assets/parabola.png", "./assets/triangle.png"]

# Verify images exist and convert to base64
images_base64 = []
for img_path in image_paths:
    if os.path.isfile(img_path):
        images_base64.append(get_image_as_base64(img_path))
    else:
        st.error(f"Image {img_path} not found in the directory.")
        st.stop()

# VALENCE AROUSAL MODELS
model_path_valence = 'model_valence.pth'
model_path_arousal = 'model_arousal.pth'

def filter_genre(df, genre):
  if genre == "blues" or genre == "jazz":
    filtered_df = df[df["genre"] == "blues"]
  elif genre == "raggae" or genre == "classical":
    filtered_df = df
  else:
    filtered_df = df[df["genre"] == genre]
  return filtered_df

# Function to display a 2x2 grid of clickable images
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
            button_key = f'click_{idx + 1}'
            if st.button('Click', key=button_key):
                st.session_state.image_clicked[idx] = True
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

            st.write(f"Image {index+1} clicked!")
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

                print("About to call run():")
                try:
                    fig = t1.run()
                    print("Run method completed, figure returned.")

                    if fig:
                        st.plotly_chart(fig)
                    else:
                        print('No figure was created.')

                except Exception as e:
                    print(f"An error occurred: {e}")

            except Exception as e:
                print(f"An error occurred: {e}")

def main():
    st.markdown("<h2 style='text-align: center;'>🎧 Moodify</h2>", unsafe_allow_html=True)

    audio_file = st.file_uploader("Please upload an audio file (MP3)", type=["mp3"])

    # If an audio file is uploaded, display an audio player
    if audio_file is not None:
        with open('audio_upload.mp3', 'wb') as f:
            f.write(audio_file.read())

        # Determine audio file path
        temp_audio_path = os.path.abspath('audio_upload.mp3')

        # Display the audio player for the uploaded file
        audio_title = audio_file.name.split(".")[0]  
        st.subheader(f"Audio Track: {audio_title}")
        st.audio(temp_audio_path, format='audio/mp3')

        # RUN PREDICTOR
        predictor = Predictor(model_path_valence, model_path_arousal)
        # Progress bar
        progress_text = "Emotion detection in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        # Emulating progress
        for percent_complete in range(100):
            time.sleep(0.01) 
            my_bar.progress(percent_complete + 1, text=progress_text)

        valence, arousal = predictor.predict(temp_audio_path)
        print(f"Valence: {valence}, Arousal: {arousal}")
        color = get_colormap(valence, arousal)
        print(f'Emotion Detected: {color}')

        features = extract_features(temp_audio_path)
        feature_values = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            outputs = model(feature_values)
            predicted_genre_index = outputs.argmax(dim=1).item()

        genre_mapping = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
                        5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

        genre = genre_mapping[predicted_genre_index]
        
        st.subheader(f"The predicted genre of the song is: {genre}")
        my_bar.empty()

        # Button to show the modal
        st.write("Now, pick a shape!")
        display_images(genre, valence, arousal, color, spotify_va)

# Run runner.py
if __name__ == "__main__":
    main()
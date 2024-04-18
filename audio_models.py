import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

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

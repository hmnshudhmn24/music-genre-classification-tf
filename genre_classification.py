import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Data path
DATA_DIR = "data"

# Audio parameters
SR = 22050
DURATION = 30
SAMPLES_PER_TRACK = SR * DURATION

# Convert audio to mel spectrogram
def audio_to_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# Load data
def load_data(data_path):
    X, y = [], []
    genres = os.listdir(data_path)
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        if not os.path.isdir(genre_path):
            continue
        for file in os.listdir(genre_path):
            if not file.endswith('.wav'):
                continue
            file_path = os.path.join(genre_path, file)
            mel = audio_to_mel_spectrogram(file_path)
            if mel.shape[1] < 128:
                continue
            mel = mel[:, :128]
            X.append(mel)
            y.append(genre)
    return np.array(X), np.array(y)

print("Loading data...")
X, y = load_data(DATA_DIR)
X = X[..., np.newaxis]  # Add channel dimension

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
print("Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
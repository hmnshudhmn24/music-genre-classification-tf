# 🎵 Music Genre Classification using TensorFlow

This project demonstrates how to classify music into genres using spectrogram images and Convolutional Neural Networks (CNN) in TensorFlow.

## 🔍 Overview

- Converts audio clips into Mel spectrogram images
- Trains a CNN on spectrograms to classify genres
- Uses GTZAN dataset structure (10 genres)

## 📦 Dependencies

```bash
pip install tensorflow librosa numpy matplotlib scikit-learn
```

## 🚀 How to Run

1. Organize dataset in folders (e.g., `data/genre_name/*.wav`)
2. Run the script:
```bash
python genre_classification.py
```

## 📁 Files

- `genre_classification.py`: Main Python script
- `README.md`: Project documentation

## 💡 Notes

- The dataset should be structured like the GTZAN dataset: one folder per genre containing `.wav` files.
- Spectrogram images are generated on the fly during training.

## 📚 Reference

- GTZAN dataset format
- TensorFlow CNNs for image classification
- Librosa for audio processing
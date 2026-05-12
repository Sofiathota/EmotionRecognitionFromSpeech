import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
import tempfile

model = tf.keras.models.load_model("emotion_model.h5")

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

def extract_mfcc(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_scaled = np.mean(mfcc.T, axis=0)

    return mfcc_scaled

st.title("🎤 Emotion Recognition From Speech")

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    features = extract_mfcc(temp_path)

    features = features.reshape(1, 40, 1)

    prediction = model.predict(features)

    predicted_class = np.argmax(prediction)

    emotion = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    st.success(f"Predicted Emotion: {emotion}")
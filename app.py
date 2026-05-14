import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
import tempfile
from keras.models import load_model

# Page Config
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

# Custom Colors & Styling
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #e0eafc, #cfdef3);
}

.main-title {
    text-align: center;
    color: #4B0082;
    font-size: 42px;
    font-weight: bold;
}

.sub-text {
    text-align: center;
    color: #333333;
    font-size: 18px;
}

.prediction-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #0B6623;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# Load Model
model = load_model("fixed_emotion_model.keras", compile=False)
# Load Label Encoder
with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# MFCC Feature Extraction
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

# Title
st.markdown(
    "<div class='main-title'>🎤 Speech Emotion Recognition</div>",
    unsafe_allow_html=True
)

# Description
st.markdown(
    "<div class='sub-text'>Deep Learning based Emotion Detection using CNN & MFCC Features</div>",
    unsafe_allow_html=True
)

st.write("")

# Upload Box
uploaded_file = st.file_uploader(
    "📂 Upload WAV Audio File",
    type=["wav"]
)

# Prediction
if uploaded_file is not None:

    st.audio(uploaded_file, format='audio/wav')

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

    st.markdown(
        f"<div class='prediction-box'>🎯 Predicted Emotion: {emotion.upper()}</div>",
        unsafe_allow_html=True
    )

# Footer
st.write("")
st.markdown("---")

st.markdown(
    "<p style='text-align:center;color:#555;'>Developed using Streamlit, TensorFlow & Librosa</p>",
    unsafe_allow_html=True
)
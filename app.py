import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import joblib
import os
from scipy.io import wavfile
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎭",
    layout="wide"
)

# Load the saved models and objects
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_emotion_lstm_model.h5')
    scaler = joblib.load('feature_scaler_lstm.pkl')
    label_encoder = joblib.load('label_encoder_lstm.pkl')
    return model, scaler, label_encoder

# Feature extraction function (same as in your training code)
def extract_features(audio, sr):
    # Initialize feature list
    features = []
    
    # 1. MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features.append(mfccs.T)
    
    # 2. Delta MFCCs
    delta_mfccs = librosa.feature.delta(mfccs)
    features.append(delta_mfccs.T)
    
    # 3. Delta-Delta MFCCs
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features.append(delta2_mfccs.T)
    
    # 4. Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.append(chroma.T)
    
    # 5. Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.append(contrast.T)
    
    # 6. Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    features.append(tonnetz.T)
    
    # 7. Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db.T)
    
    # 8. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(zcr.T)
    
    # 9. Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(rolloff.T)
    
    # 10. Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(centroid.T)
    
    # 11. RMS Energy
    rms = librosa.feature.rms(y=audio)
    features.append(rms.T)
    
    # Concatenate all features
    combined_features = np.concatenate(features, axis=1)
    
    # Pad or truncate to fixed length (130 time steps)
    max_len = 130
    if combined_features.shape[0] < max_len:
        pad_width = max_len - combined_features.shape[0]
        combined_features = np.pad(combined_features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        combined_features = combined_features[:max_len, :]
        
    return combined_features

# Prediction function
def predict_emotion(audio_file, model, scaler, label_encoder):
    # Load audio file
    audio, sr = librosa.load(audio_file, duration=3, sr=22050)
    
    # Extract features
    features = extract_features(audio, sr)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Reshape for model
    features_reshaped = np.expand_dims(features_scaled, axis=0)
    
    # Predict
    prediction = model.predict(features_reshaped)
    predicted_emotion = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    emotion_probabilities = prediction[0]
    
    return predicted_emotion, emotion_probabilities, label_encoder.classes_

# Main Streamlit app
def main():
    st.title("🎭 Speech Emotion Recognition")
    st.write("Upload an audio file to detect the emotion in the speech.")
    
    # Load models
    try:
        model, scaler, label_encoder = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File uploader
    audio_file = st.file_uploader("Upload an audio file", type=["wav"])
    
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        
        # Create a button to trigger prediction
        if st.button("Detect Emotion"):
            with st.spinner("Analyzing audio..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Make prediction
                    predicted_emotion, probabilities, emotion_labels = predict_emotion(
                        tmp_file_path, model, scaler, label_encoder
                    )
                    
                    # Delete temporary file
                    os.unlink(tmp_file_path)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"Detected Emotion: {predicted_emotion.upper()}")
                        
                        # Create probability bar chart
                        import plotly.graph_objects as go
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=probabilities * 100,
                                y=emotion_labels,
                                orientation='h',
                                marker_color='rgba(50, 171, 96, 0.6)',
                            )
                        ])
                        
                        fig.update_layout(
                            title="Emotion Probabilities",
                            xaxis_title="Probability (%)",
                            yaxis_title="Emotion",
                            width=600,
                            height=400
                        )
                        
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Display confidence scores
                        st.subheader("Confidence Scores:")
                        for emotion, prob in zip(emotion_labels, probabilities):
                            st.write(f"{emotion}: {prob*100:.2f}%")
                            
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    # Add information about the model
    with st.expander("ℹ️ About the Model"):
        st.write("""
        This emotion recognition model uses a Bidirectional LSTM architecture to classify speech emotions.
        It can detect the following emotions:
        - Neutral
        - Calm
        - Happy
        - Sad
        - Angry
        - Fearful
        - Disgust
        - Surprised
        
        The model analyzes various audio features including MFCCs, spectral features, and pitch information
        to make predictions.
        """)
    
    # Add footer
    st.markdown("""
    ---
    Created with ❤️ using Streamlit | Model: BiLSTM with Attention
    """)

if __name__ == "__main__":
    main()

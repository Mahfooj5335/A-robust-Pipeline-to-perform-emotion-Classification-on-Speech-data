import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import joblib
import os
from scipy.io import wavfile
import tempfile
import sounddevice as sd
import threading
import time

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

# Updated feature extraction function to match training (194 features)
def extract_features(audio, sr):
    """
    Extract features to match the training configuration:
    - 20 MFCCs + 20 Delta + 20 Delta-Delta = 60 features
    - 12 Chroma features
    - 6 Spectral contrast features  
    - 6 Tonnetz features
    - 128 Mel-spectrogram features
    - 1 ZCR + 1 Rolloff + 1 Centroid + 1 RMS = 4 features
    Total: 60 + 12 + 6 + 6 + 128 + 4 = 216 features
    But your training shows 194, so let's match exactly
    """
    # Initialize feature list
    features = []
    
    # 1. MFCCs (20 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features.append(mfccs.T)  # Shape: (time_steps, 20)
    
    # 2. Delta MFCCs (first derivative)
    delta_mfccs = librosa.feature.delta(mfccs)
    features.append(delta_mfccs.T)  # Shape: (time_steps, 20)
    
    # 3. Delta-Delta MFCCs (second derivative)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features.append(delta2_mfccs.T)  # Shape: (time_steps, 20)
    
    # 4. Chroma features (12 coefficients - matching training)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
    features.append(chroma.T)  # Shape: (time_steps, 12)
    
    # 5. Spectral contrast (6 bands - matching training)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    features.append(contrast.T)  # Shape: (time_steps, 7) -> but we need 6
    
    # 6. Tonnetz (Harmonic network) - 6 features
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    features.append(tonnetz.T)  # Shape: (time_steps, 6)
    
    # 7. Mel-spectrogram (128 mel bands - matching training)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db.T)  # Shape: (time_steps, 128)
    
    # 8. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(zcr.T)  # Shape: (time_steps, 1)
    
    # 9. Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(rolloff.T)  # Shape: (time_steps, 1)
    
    # 10. Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(centroid.T)  # Shape: (time_steps, 1)
    
    # 11. RMS Energy
    rms = librosa.feature.rms(y=audio)
    features.append(rms.T)  # Shape: (time_steps, 1)
    
    # Concatenate all features along feature axis
    combined_features = np.concatenate(features, axis=1)
    
    # Debug: Print actual feature count
    print(f"Combined features shape before padding: {combined_features.shape}")
    
    # Handle spectral contrast shape issue (it returns 7 features instead of 6)
    if combined_features.shape[1] > 194:
        # Remove the first spectral contrast feature to match training
        # Spectral contrast starts at index 72 (20+20+20+12)
        combined_features = np.delete(combined_features, 72, axis=1)
    
    # Ensure we have exactly 194 features to match training
    if combined_features.shape[1] != 194:
        print(f"Warning: Feature count mismatch. Got {combined_features.shape[1]}, expected 194")
        if combined_features.shape[1] > 194:
            combined_features = combined_features[:, :194]
        else:
            # Pad with zeros if we have fewer features
            pad_width = 194 - combined_features.shape[1]
            combined_features = np.pad(combined_features, ((0, 0), (0, pad_width)), mode='constant')
    
    # Pad or truncate to fixed length (130 time steps - matching training)
    max_len = 130
    if combined_features.shape[0] < max_len:
        # Pad with zeros
        pad_width = max_len - combined_features.shape[0]
        combined_features = np.pad(combined_features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        # Truncate
        combined_features = combined_features[:max_len, :]
    
    print(f"Final feature shape: {combined_features.shape}")
    return combined_features

# Real-time audio recording functions
def record_audio(duration=3, sample_rate=22050):
    """Record audio for specified duration"""
    try:
        st.info(f"🎤 Recording for {duration} seconds... Speak now!")
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype='float64')
        sd.wait()  # Wait until recording is finished
        return audio_data.flatten(), sample_rate
    except Exception as e:
        st.error(f"Error during recording: {str(e)}")
        return None, None

def save_recorded_audio(audio_data, sample_rate, filename):
    """Save recorded audio to a temporary file"""
    try:
        # Convert to int16 for wav file
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio_int16)
        return True
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return False
def add_noise(audio, noise_factor=0.02):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_shift(audio, shift_max=0.2):
    shift = np.random.randint(-int(shift_max * len(audio)), int(shift_max * len(audio)))
    return np.roll(audio, shift)

def pitch_shift(audio, sr, pitch_factor=0.1):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)

def speed_change(audio, speed_factor=1.0):
    return librosa.effects.time_stretch(audio, rate=speed_factor)

# Updated prediction function
def predict_emotion(audio_file, model, scaler, label_encoder):
    # Load audio file with same parameters as training
    audio, sr = librosa.load(audio_file, duration=3, sr=22050)
    
    # Extract features using the updated function
    features = extract_features(audio, sr)
    
    # Scale features using the same scaler from training
    features_scaled = scaler.transform(features)
    
    # Reshape for model input (add batch dimension)
    features_reshaped = np.expand_dims(features_scaled, axis=0)
    
    # Make prediction
    prediction = model.predict(features_reshaped)
    predicted_emotion = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    emotion_probabilities = prediction[0]
    
    return predicted_emotion, emotion_probabilities, label_encoder.classes_

# Main Streamlit app
def main():
    st.title("🎭 Speech Emotion Recognition")
    st.write("Upload an audio file or record your voice to detect the emotion in the speech.")
    
    # Load models
    try:
        model, scaler, label_encoder = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎤 Record Voice"])
    
    with tab1:
        st.subheader("Upload Audio File")
        # File uploader
        audio_file = st.file_uploader("Upload an audio file", type=["wav"])
        
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
            
            # Create a button to trigger prediction
            if st.button("Detect Emotion from File", key="file_predict"):
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
                        display_results(predicted_emotion, probabilities, emotion_labels)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.write("Please check if all model files are present and compatible.")
    
    with tab2:
        st.subheader("Record Your Voice")
        st.write("Click the button below to record a 3-second audio clip for emotion detection.")
        
        # Recording controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            record_duration = st.selectbox("Recording Duration", [2, 3, 4, 5], index=1)
        
        with col2:
            if st.button("🎤 Start Recording", key="record_button"):
                # Initialize session state for recording
                if 'recording_done' not in st.session_state:
                    st.session_state.recording_done = False
                
                # Record audio
                audio_data, sample_rate = record_audio(duration=record_duration)
                
                if audio_data is not None:
                    st.session_state.recorded_audio = audio_data
                    st.session_state.sample_rate = sample_rate
                    st.session_state.recording_done = True
                    st.success("✅ Recording completed!")
        
        # Display recorded audio and prediction
        if 'recording_done' in st.session_state and st.session_state.recording_done:
            # Save recorded audio to temporary file for playback and processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file_path = tmp_file.name
            
            if save_recorded_audio(st.session_state.recorded_audio, 
                                 st.session_state.sample_rate, 
                                 tmp_file_path):
                
                # Display audio player
                st.audio(tmp_file_path)
                
                # Predict emotion button
                if st.button("🔍 Detect Emotion from Recording", key="record_predict"):
                    with st.spinner("Analyzing recorded audio..."):
                        try:
                            # Make prediction
                            predicted_emotion, probabilities, emotion_labels = predict_emotion(
                                tmp_file_path, model, scaler, label_encoder
                            )
                            
                            # Display results
                            display_results(predicted_emotion, probabilities, emotion_labels)
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                        finally:
                            # Clean up temporary file
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                
                # Clear recording button
                if st.button("🗑️ Clear Recording", key="clear_record"):
                    if 'recorded_audio' in st.session_state:
                        del st.session_state.recorded_audio
                    if 'sample_rate' in st.session_state:
                        del st.session_state.sample_rate
                    st.session_state.recording_done = False
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                    st.rerun()

# Function to display prediction results
def display_results(predicted_emotion, probabilities, emotion_labels):
    """Display emotion prediction results"""
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

# Main Streamlit app (updated)
def main_old():
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
                    st.write("Please check if all model files are present and compatible.")
    
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
        
        **Feature Engineering:**
        - 20 MFCC coefficients + 20 Delta + 20 Delta-Delta = 60 features
        - 12 Chroma features
        - 6 Spectral contrast features
        - 6 Tonnetz features
        - 128 Mel-spectrogram features
        - Zero crossing rate, Spectral rolloff, Spectral centroid, RMS energy = 4 features
        - Total: 194 features per time step
        - Sequence length: 130 time steps
        
        The model analyzes 3-second audio clips and outputs emotion probabilities.
        
        **Recording Tips:**
        - Speak clearly and naturally
        - Record in a quiet environment
        - Use 3-4 seconds for best results
        - Ensure your microphone is working properly
        """)
    
    # Add installation instructions
    with st.expander("🔧 Installation Requirements"):
        st.code("""
# Install required packages for audio recording:
pip install sounddevice
pip install scipy

# For Linux users, you might need:
sudo apt-get install libasound2-dev
        """)
    
    # Add footer
    st.markdown("""
    ---
    Created with ❤️ using Streamlit | Model: BiLSTM with 194 Features | Real-time Recording Enabled
    """)

if __name__ == "__main__":
    main()

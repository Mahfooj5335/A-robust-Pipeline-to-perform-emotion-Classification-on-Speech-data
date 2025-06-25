import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import joblib
import os
from scipy.io import wavfile
import tempfile
# New imports for voice recording
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
import pydub
import logging
from datetime import datetime

# [Previous imports and functions remain the same until main()]

# New function to convert audio samples to wav file
def convert_audio_to_wav(audio_frames):
    # Convert audio frames to wav file
    audio_segments = []
    for audio_frame in audio_frames:
        sound = pydub.AudioSegment(
            data=audio_frame.to_ndarray().tobytes(),
            sample_width=audio_frame.format.bytes,
            frame_rate=audio_frame.sample_rate,
            channels=len(audio_frame.layout.channels),
        )
        audio_segments.append(sound)
    
    final_audio = sum(audio_segments)
    
    # Export as wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        final_audio.export(tmp_file.name, format="wav")
        return tmp_file.name

def main():
    st.title("🎭 Speech Emotion Recognition")
    st.write("Upload an audio file or record your voice to detect emotion in the speech.")
    
    # Load models
    try:
        model, scaler, label_encoder = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Create tabs for upload and recording
    tab1, tab2 = st.tabs(["Upload Audio", "Record Voice"])
    
    with tab1:
        # Original file upload code
        audio_file = st.file_uploader("Upload an audio file", type=["wav"])
        
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
            
            if st.button("Detect Emotion (Upload)"):
                process_audio_file(audio_file, model, scaler, label_encoder)
    
    with tab2:
        # Voice recording section
        st.write("Click 'Start' to begin recording your voice")
        
        # Webrtc component for recording
        audio_frames = []
        
        def audio_callback(frame):
            audio_frames.append(frame)
            return frame
        
        webrtc_ctx = webrtc_streamer(
            key="voice-recorder",
            mode=WebRtcMode.RECORDING,
            audio_receiver_size=1024,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
            on_audio_frame=audio_callback,
        )
        
        if not webrtc_ctx.state.playing and len(audio_frames) > 0:
            st.write("Recording finished! Processing audio...")
            # Convert the recorded audio to wav file
            wav_file_path = convert_audio_to_wav(audio_frames)
            st.audio(wav_file_path, format="audio/wav")
            
            if st.button("Detect Emotion (Recording)"):
                process_audio_file(wav_file_path, model, scaler, label_encoder)
                # Clean up temporary file
                os.unlink(wav_file_path)
            
            # Clear the frames for next recording
            audio_frames.clear()

def process_audio_file(audio_file, model, scaler, label_encoder):
    """Helper function to process audio and show results"""
    with st.spinner("Analyzing audio..."):
        try:
            # Handle both uploaded and recorded files
            if isinstance(audio_file, str):
                # For recorded audio (file path)
                tmp_file_path = audio_file
                delete_after = False
            else:
                # For uploaded audio (file object)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_file_path = tmp_file.name
                delete_after = True
            
            # Make prediction
            predicted_emotion, probabilities, emotion_labels = predict_emotion(
                tmp_file_path, model, scaler, label_encoder
            )
            
            # Delete temporary file if needed
            if delete_after:
                os.unlink(tmp_file_path)
            
            # Display results
            display_results(predicted_emotion, probabilities, emotion_labels)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Please check if the audio file is valid and try again.")

def display_results(predicted_emotion, probabilities, emotion_labels):
    """Helper function to display prediction results"""
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

    # [Rest of your code remains the same]

if __name__ == "__main__":
    main()

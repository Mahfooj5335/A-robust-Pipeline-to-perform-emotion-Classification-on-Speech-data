import numpy as np
import tensorflow as tf
import joblib
import sys
import os
import librosa
import pandas as pd

# ================== CONFIG ==================
MODEL_PATH = 'best_emotion_lstm_model.h5'
SCALER_PATH = 'feature_scaler_lstm.pkl'
LABEL_ENCODER_PATH = 'label_encoder_lstm.pkl'
TEST_CSV_PATH = 'test_files.csv'  # CSV should have a column 'filepath' pointing to .wav files

# ================== LOAD OBJECTS ==================
def load_all():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, scaler, label_encoder

# ================== FEATURE EXTRACTION ==================
def extract_features(audio, sr):
    features = []
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features.append(mfccs.T)
    delta_mfccs = librosa.feature.delta(mfccs)
    features.append(delta_mfccs.T)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features.append(delta2_mfccs.T)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.append(chroma.T)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.append(contrast.T)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    features.append(tonnetz.T)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db.T)
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(zcr.T)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(rolloff.T)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(centroid.T)
    rms = librosa.feature.rms(y=audio)
    features.append(rms.T)
    combined_features = np.concatenate(features, axis=1)
    max_len = 130
    if combined_features.shape[0] < max_len:
        pad_width = max_len - combined_features.shape[0]
        combined_features = np.pad(combined_features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        combined_features = combined_features[:max_len, :]
    return combined_features

# ================== PREDICT FUNCTION ==================
def predict_audio(filepath, model, scaler, label_encoder):
    try:
        audio, sr = librosa.load(filepath, duration=3, sr=22050)
        features = extract_features(audio, sr)
        features_scaled = scaler.transform(features)
        features_scaled = np.expand_dims(features_scaled, axis=0)
        pred = model.predict(features_scaled)
        pred_idx = np.argmax(pred, axis=1)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        conf = pred[0][pred_idx]
        return pred_label, conf
    except Exception as e:
        return f"ERROR: {e}", 0.0

# ================== MAIN FUNCTION ==================
def main():
    # Load model, scaler, encoder
    model, scaler, label_encoder = load_all()
    print("Loaded model, scaler, and label encoder.")

    # Load test data CSV
    if not os.path.isfile(TEST_CSV_PATH):
        print(f"Test file list {TEST_CSV_PATH} not found. Please provide a CSV with a 'filepath' column.")
        sys.exit(1)
    test_df = pd.read_csv(TEST_CSV_PATH)
    if 'filepath' not in test_df.columns:
        print("CSV must have a 'filepath' column.")
        sys.exit(1)
    print(f"Testing on {len(test_df)} files.")

    preds = []
    confs = []
    for i, row in test_df.iterrows():
        filepath = row['filepath']
        if not os.path.isfile(filepath):
            print(f"File not found: {filepath}")
            preds.append("ERROR")
            confs.append(0.0)
            continue
        pred_label, conf = predict_audio(filepath, model, scaler, label_encoder)
        preds.append(pred_label)
        confs.append(conf)
        print(f"[{i+1}/{len(test_df)}] {os.path.basename(filepath)} --> {pred_label} ({conf:.2f})")

    test_df['predicted_emotion'] = preds
    test_df['confidence'] = confs

    # Save results
    out_csv = "test_results.csv"
    test_df.to_csv(out_csv, index=False)
    print(f"\nTest results saved to {out_csv}")

if __name__ == '__main__':
    main()

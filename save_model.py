import joblib
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Save the model objects (assuming you have already trained the model)
def save_model_objects(model, scaler, label_encoder):
    # Save the LSTM model
    model.save('emotion_model/best_emotion_lstm_model.h5')
    
    # Save the scaler
    joblib.dump(scaler, 'emotion_model/feature_scaler_lstm.pkl')
    
    # Save the label encoder
    joblib.dump(label_encoder, 'emotion_model/label_encoder_lstm.pkl')
    
    print("All model objects saved successfully!")

save_model_objects(model, scaler, label_encoder)

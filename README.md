Project : Enhanced Speech Emotion Recognition using RAVDESS Dataset

1. Project Overview :

This project implements a comprehensive **Speech Emotion Recognition (SER)** system using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The system employs advanced deep learning techniques, particularly **LSTM neural networks**, combined with sophisticated audio feature extraction and data augmentation to achieve high-accuracy emotion classification.

- **Advanced Feature Extraction**: 194-dimensional feature vectors including MFCCs, Delta coefficients, Chroma, Spectral contrast, and more
- **Data Augmentation**: Noise addition, time shifting, pitch shifting, and speed changes to improve model robustness
- **Deep Learning Architecture**: Bidirectional LSTM with batch normalization and dropout for optimal performance
- **Comprehensive Evaluation**: Multiple baseline models comparison with detailed metrics
- **High Performance**: Achieves >80% accuracy and F1-score requirements

 📁 Dataset Structure

RAVDESS Dataset Format
The dataset contains emotional speech and song recordings from 24 professional actors (12 female, 12 male):

```
Audio_Speech_Actors_01-24/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-01-01-01-02-01.wav
│   └── ...
├── Actor_02/
└── ...

Audio_Song_Actors_01-24/
├── Actor_01/
├── Actor_02/
└── ...
```

2 .Filename Encoding
Each audio file follows the pattern: `03-02-06-01-02-01-12.wav`

| Position | Description | Values |
|----------|-------------|---------|
| 1 | Modality | 03 = audio-only |
| 2 | Vocal Channel | 01 = speech, 02 = song |
| 3 | Emotion | 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised |
| 4 | Intensity | 01=normal, 02=strong |
| 5 | Statement | 01="Kids are talking by the door", 02="Dogs are sitting by the door" |
| 6 | Repetition | 01=1st repetition, 02=2nd repetition |
| 7 | Actor | 01-24 (odd=male, even=female) |

 3. Technical Architecture

 Feature Extraction Pipeline

The system extracts **194 audio features** per file:

3.1. **MFCC Features (20 coefficients)**
   - Standard Mel-Frequency Cepstral Coefficients
   - Delta MFCCs (1st derivative) - 20 features
   - Delta-Delta MFCCs (2nd derivative) - 20 features

3.2. **Chroma Features (12 coefficients)**
   - Pitch class profiles representing harmonic content

3.3. **Spectral Features**
   - Spectral Contrast (6 bands)
   - Spectral Centroid
   - Spectral Rolloff

3.4. **Advanced Features**
   - Tonnetz (6 coefficients) - Harmonic network analysis
   - Mel-spectrogram (128 coefficients)
   - Zero Crossing Rate
   - RMS Energy

4. Data Augmentation Techniques

- **Noise Addition**: Gaussian noise with configurable factor
- **Time Shifting**: Temporal displacement of audio signals
- **Pitch Shifting**: Frequency modification using librosa
- **Speed Change**: Temporal stretching/compression

5 . LSTM Model Architecture

```python
Model: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional (Bidirectional) (None, 130, 256)        332,800   
batch_normalization          (None, 130, 256)         1,024     
bidirectional_1 (Bidirection) (None, 130, 128)        164,352   
batch_normalization_1        (None, 130, 128)         512       
lstm_2 (LSTM)               (None, 32)                20,608    
batch_normalization_2        (None, 32)               128       
dense (Dense)               (None, 64)                2,112     
dropout (Dropout)           (None, 64)                0         
dense_1 (Dense)             (None, 32)                2,080     
dropout_1 (Dropout)         (None, 32)                0         
dense_2 (Dense)             (None, 8)                 264       
=================================================================
Total params: 523,880
Trainable params: 523,048
Non-trainable params: 832
```


5. Usage Instructions

5.1 . Dataset Preparation

Place your RAVDESS dataset in the following structure:
```
/path/to/dataset/
├── Audio_Speech_Actors_01-24/
└── Audio_Song_Actors_01-24/
```

5.2 . Update Dataset Paths

Modify the paths in the code:
```python
speech_path = "/your/path/to/Audio_Speech_Actors_01-24"
song_path = "/your/path/to/Audio_Song_Actors_01-24"
```

5.3 . Run the Complete Pipeline

```bash
python emotion_recognition.py
```

The script will automatically:
- Load and parse the dataset
- Extract advanced audio features
- Apply data augmentation
- Train the LSTM model
- Evaluate performance against baseline models
- Generate comprehensive visualizations

6.  Performance Metrics

6.1 Evaluation Criteria

The model is evaluated against stringent requirements:
- ✅ **Overall Accuracy**: >80%
- ✅ **F1-Score**: >80%
- ✅ **Per-Class Accuracy**: >75% for each emotion
- ✅ **Confusion Matrix**: Detailed class-wise performance



7.  Per-Emotion Performance

Detailed Classification Report (Test Set):
              precision    recall  f1-score   support

       angry       1.00      0.93      0.96       170
        calm       1.00      1.00      1.00       169
     disgust       1.00      1.00      1.00        87
     fearful       1.00      1.00      1.00       169
       happy       1.00      1.00      1.00       170
     neutral       1.00      1.00      1.00        84
         sad       1.00      1.00      1.00       169
   surprised       0.88      1.00      0.93        86
...
sad: 1.0000
surprised: 1.0000

8.  Model performance
   Model                     Accuracy     F1 Score     Status
------------------------------------------------------------
LSTM (Deep Learning)      0.9891       0.9893       ✓ PASS
Random Forest             1.0000       1.0000       ✓ PASS
SVM                       0.9982       0.9982       ✓ PASS

Best Model: Random Forest (F1: 1.0000)

Overall Requirements Check:
  Overall Accuracy: 0.9891 (✓ >80%)
  F1 Score: 0.9893 (✓ >80%)
  Min Class Accuracy: 0.9294 (✓ >75%)
  All Requirements Met: ✓ YES





9. Audio Processing Parameters
- **Sample Rate**: 22,050 Hz
- **Duration**: 3 seconds per clip
- **Feature Extraction Window**: Hamming window with 50% overlap
- **MFCC Configuration**: 20 coefficients, 128 mel filters

10. Training Configuration
- **Batch Size**: 32
- **Learning Rate**: 0.001 (with adaptive reduction)
- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Regularization**: Dropout (0.3-0.5), L2 regularization
- **Early Stopping**: Patience=15 epochs on validation loss

11. Data Augmentation Parameters
- **Noise Factor**: 0.02
- **Time Shift**: ±20% of audio length
- **Pitch Shift**: ±2 semitones
- **Speed Change**: 0.8x to 1.2x original speed

12.  Applications

This emotion recognition system can be applied in various domains:

- **Healthcare**: Mental health monitoring and therapy assistance
- **Education**: Student engagement assessment in e-learning
- **Customer Service**: Automated emotion detection in call centers
- **Entertainment**: Emotion-aware content recommendation systems
- **Human-Computer Interaction**: Adaptive user interfaces
- **Market Research**: Consumer sentiment analysis from voice data


13.  Acknowledgments

- **RAVDESS Dataset**: Livingstone & Russo (2018) for providing the comprehensive emotional speech dataset
- **Librosa**: For excellent audio processing capabilities
- **TensorFlow/Keras**: For deep learning framework support
- **Scikit-learn**: For machine learning utilities and baseline models

14.  References

1. Livingstone, S.R. & Russo, F.A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.
2. McFee, B., et al. (2015). librosa: Audio and music signal analysis in python. Proceedings of the 14th python in science conference.
3. Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.


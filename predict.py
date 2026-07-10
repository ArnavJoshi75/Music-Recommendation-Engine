import os
import sys
import numpy as np
# pyrefly: ignore [missing-import]
import librosa
import tensorflow as tf

def extract_features(audio_path, sr=22050, duration=30, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extracts Mel-spectrogram features from an audio file, matching the logic
    used during training in data_generator.py.
    """
    target_length = int(sr * duration)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Zero-pad if the track is too short, or trim if too long
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    # Compute Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # Convert power to decibels (log scale)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Transpose so time is the first axis: (time_steps, n_mels)
    S_dB = S_dB.T
    
    # Add channel dimension: (time_steps, n_mels, 1)
    S_dB = np.expand_dims(S_dB, axis=-1)
    
    return np.array([S_dB]) # Return as batch of 1: shape (1, time_steps, n_mels, 1)

def predict_genre(audio_path, model_path='best_crnn_model.keras'):
    if not os.path.exists(audio_path):
        print(f"Error: Could not find audio file at {audio_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Could not find model file at {model_path}. Please train the model first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Extracting features from {audio_path}...")
    X = extract_features(audio_path)
    
    print("Predicting...")
    predictions = model.predict(X, verbose=0)[0]
    predicted_index = np.argmax(predictions)
    
    # The FMA 'small' dataset has these 8 genres. 
    # Since LabelEncoder sorts them alphabetically, this is their expected order.
    genres = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 
              'Instrumental', 'International', 'Pop', 'Rock']
    
    print("\n" + "="*30)
    print("         PREDICTION         ")
    print("="*30)
    
    if len(genres) == len(predictions):
        predicted_genre = genres[predicted_index]
        print(f"Predicted Genre: {predicted_genre} (Confidence: {predictions[predicted_index]:.2%})")
        print("\nAll Probabilities:")
        
        # Sort predictions by confidence in descending order
        sorted_indices = np.argsort(predictions)[::-1]
        for idx in sorted_indices:
            print(f"  {genres[idx]}: {predictions[idx]:.2%}")
    else:
        print(f"Predicted Class Index: {predicted_index} (Confidence: {predictions[predicted_index]:.2%})")
    
    print("="*30 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
        print("Example: python predict.py sample.mp3")
    else:
        audio_file = sys.argv[1]
        predict_genre(audio_file)

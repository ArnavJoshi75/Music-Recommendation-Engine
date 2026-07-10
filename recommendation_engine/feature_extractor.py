"""
feature_extractor.py
====================
Model surgery module for the CRNN Music Genre Classifier.

Loads the fully-trained Keras model and creates a "headless" version
by bypassing the final Softmax classification layer. The output of the
penultimate Dense(64, relu) layer serves as a 64-dimensional latent
embedding that captures the musical "fingerprint" of an audio track.

Exports:
    load_headless_model(model_path) -> tf.keras.Model
    extract_features(audio_path)    -> np.ndarray  (1, 1292, 128, 1)
"""

import os
import numpy as np
import librosa
import tensorflow as tf


# ---------------------------------------------------------------------------
# Audio preprocessing constants (must match training pipeline exactly)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
DURATION = 30          # seconds
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def extract_features(audio_path: str) -> np.ndarray:
    """
    Extracts a Mel-spectrogram from a raw audio file, replicating the
    exact preprocessing used during CRNN training.

    Pipeline:
        .mp3 -> librosa.load (mono, 22050 Hz, 30 s)
             -> zero-pad / trim to exactly 30 s
             -> Mel-spectrogram (128 bands)
             -> power_to_db (log scale)
             -> transpose to (time_steps, n_mels)
             -> add channel dim -> (time_steps, n_mels, 1)
             -> add batch dim   -> (1, time_steps, n_mels, 1)

    Args:
        audio_path: Absolute or relative path to an audio file.

    Returns:
        Numpy array of shape (1, 1292, 128, 1) — a single-sample batch
        ready for model.predict().
    """
    target_length = int(SAMPLE_RATE * DURATION)

    # Load raw audio as a mono waveform
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

    # Enforce exact length: pad short tracks, trim long ones
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    # Compute the Mel-spectrogram (power spectrum)
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    # Convert power to decibels (log scale) — matches training exactly
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Transpose: (n_mels, time_steps) -> (time_steps, n_mels)
    S_dB = S_dB.T

    # Add channel dimension: (time_steps, n_mels, 1)
    S_dB = np.expand_dims(S_dB, axis=-1)

    # Add batch dimension: (1, time_steps, n_mels, 1)
    return np.expand_dims(S_dB, axis=0).astype(np.float32)


def load_headless_model(model_path: str) -> tf.keras.Model:
    """
    Performs "model surgery" on the trained CRNN.

    The original architecture ends with:
        ... -> Dense(64, relu)   <- layer[-3]  (penultimate — OUR EMBEDDING)
            -> Dropout(0.3)      <- layer[-2]
            -> Dense(8, softmax) <- layer[-1]  (classification head)

    We create a new Model that shares the same weights but outputs the
    64-dimensional embedding from the penultimate Dense layer, effectively
    discarding the classification head.

    Args:
        model_path: Path to the saved .keras model file.

    Returns:
        A tf.keras.Model that maps:
            input spectrogram (1, 1292, 128, 1) -> embedding (1, 64)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please ensure best_crnn_model.keras exists."
        )

    print(f"[FeatureExtractor] Loading full model from: {model_path}")
    full_model = tf.keras.models.load_model(model_path)

    # Identify the penultimate Dense(64, relu) layer
    # Architecture: ..., Dense(64), Dropout(0.3), Dense(8)
    embedding_layer = full_model.layers[-3]

    # Keras 3 (TF 2.16+) removed layer.output_shape; use layer.output.shape
    try:
        emb_shape = embedding_layer.output.shape
    except AttributeError:
        emb_shape = embedding_layer.output_shape
    print(f"[FeatureExtractor] Embedding layer: '{embedding_layer.name}' "
          f"-> output shape {emb_shape}")

    # Build a new model that shares weights but stops at the embedding layer
    headless_model = tf.keras.Model(
        inputs=full_model.inputs,
        outputs=embedding_layer.output,
        name="crnn_feature_extractor"
    )

    # Freeze all weights — we never want gradient updates
    for layer in headless_model.layers:
        layer.trainable = False

    print(f"[FeatureExtractor] Headless model ready. "
          f"Output: {headless_model.output.shape}")

    return headless_model


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Default model path (assumes running from project root)
    default_model = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "best_crnn_model.keras"
    )

    model = load_headless_model(default_model)
    model.summary()

    # Optionally test with an audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        features = extract_features(audio_file)
        print(f"\nSpectrogram shape: {features.shape}")
        embedding = model.predict(features, verbose=0)
        print(f"Embedding shape:   {embedding.shape}")
        print(f"Embedding vector:  {embedding[0][:8]}... (first 8 dims)")

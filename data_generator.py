import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def save_spectrogram_plot(spectrogram, genre_label, save_dir="visualizations/data"):
    """Saves a Mel-Spectrogram as an image file."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    # Shape is (time_steps, n_mels, 1) -> (n_mels, time_steps)
    S_dB = spectrogram[:, :, 0].T 
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=22050, hop_length=512, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram (Genre: {genre_label})')
    plt.tight_layout()
    filename = f"spectrogram_{genre_label.replace('/', '_')}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Saved spectrogram plot to {os.path.join(save_dir, filename)}")


def apply_spec_augment(spec, num_mask=2, freq_masking=0.15, time_masking=0.20):
    """
    Applies Frequency and Time masking to a spectrogram.
    spec: shape (time_steps, n_mels)
    """
    spec_aug = spec.copy()
    time_steps, n_mels = spec_aug.shape
    
    for _ in range(num_mask):
        # Frequency masking
        f_mask_max = int(freq_masking * n_mels)
        if f_mask_max > 0:
            f_mask = random.randint(1, f_mask_max)
            f0 = random.randint(0, n_mels - f_mask)
            spec_aug[:, f0:f0+f_mask] = spec_aug.mean()

        # Time masking
        t_mask_max = int(time_masking * time_steps)
        if t_mask_max > 0:
            t_mask = random.randint(1, t_mask_max)
            t0 = random.randint(0, time_steps - t_mask)
            spec_aug[t0:t0+t_mask, :] = spec_aug.mean()
        
    return spec_aug

def add_gaussian_noise(spec, noise_factor=0.05):
    """Adds random gaussian noise to the spectrogram."""
    noise = np.random.normal(0, spec.std(), spec.shape) * noise_factor
    return spec + noise


class FmaDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Data Generator for FMA Dataset.
    Dynamically loads audio files and computes Mel-spectrograms.
    """
    def __init__(self, track_ids, labels, data_dir, batch_size=32, sr=22050, duration=30, n_mels=128, n_fft=2048, hop_length=512, shuffle=True, augment=False):
        self.track_ids = track_ids
        self.labels = labels
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.track_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.track_ids[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        X, y = self.__data_generation(batch_ids, batch_labels)
        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.track_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _get_audio_path(self, track_id):
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(self.data_dir, tid_str[:3], tid_str + '.mp3')

    def __data_generation(self, batch_ids, batch_labels):
        X = []
        valid_labels = []
        target_length = int(self.sr * self.duration)

        for i, track_id in enumerate(batch_ids):
            path = self._get_audio_path(track_id)
            try:
                # Load audio
                y, sr = librosa.load(path, sr=self.sr, duration=self.duration)
                
                # Zero-pad if the track is too short
                if len(y) < target_length:
                    y = np.pad(y, (0, target_length - len(y)))
                else:
                    y = y[:target_length]

                # Compute Mel-spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
                
                # Convert power to decibels (log scale)
                S_dB = librosa.power_to_db(S, ref=np.max)
                
                # Transpose so time is the first axis: (time_steps, n_mels)
                S_dB = S_dB.T
                
                # Apply data augmentation only during training
                if self.augment:
                    if random.random() < 0.5:
                        S_dB = apply_spec_augment(S_dB)
                    if random.random() < 0.3:
                        S_dB = add_gaussian_noise(S_dB)
                
                # Add channel dimension: (time_steps, n_mels, 1)
                S_dB = np.expand_dims(S_dB, axis=-1)
                
                X.append(S_dB)
                valid_labels.append(batch_labels[i])
            except Exception as e:
                print(f"Warning: Error loading {path}: {e}")
                
        # Return as numpy arrays
        return np.array(X), np.array(valid_labels)

def load_fma_metadata(metadata_path, fma_small_dir):
    """
    Reads the FMA tracks.csv and extracts track_ids and genre_top labels for the 'small' subset.
    """
    print("Reading metadata from tracks.csv...")
    tracks = pd.read_csv(os.path.join(metadata_path, 'tracks.csv'), index_col=0, header=[0, 1], low_memory=False)
    
    # Filter for the 'small' subset
    small_subset = tracks[tracks[('set', 'subset')] == 'small']
    
    # Get genre labels
    labels = small_subset[('track', 'genre_top')]
    
    # Drop rows with missing labels
    labels = labels.dropna()
    track_ids = labels.index.tolist()
    
    print("Verifying file existence...")
    valid_ids = []
    valid_labels = []
    for tid, label in zip(track_ids, labels):
        tid_str = '{:06d}'.format(tid)
        path = os.path.join(fma_small_dir, tid_str[:3], tid_str + '.mp3')
        if os.path.exists(path):
            valid_ids.append(tid)
            valid_labels.append(label)
            
    return valid_ids, valid_labels

def prepare_data(metadata_path, fma_small_dir, batch_size=32):
    """
    Loads metadata, encodes labels, and returns train and validation generators.
    """
    track_ids, labels = load_fma_metadata(metadata_path, fma_small_dir)
    print(f"Found {len(track_ids)} valid tracks.")
    
    # Encode text labels into integers
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # One-hot encoding for categorical crossentropy
    num_classes = len(encoder.classes_)
    one_hot_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes=num_classes)
    
    # Stratified split to maintain genre balance
    X_train, X_val, y_train, y_val = train_test_split(
        track_ids, one_hot_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Detected {num_classes} classes: {encoder.classes_}")
    
    train_gen = FmaDataGenerator(X_train, y_train, fma_small_dir, batch_size=batch_size, augment=True)
    val_gen = FmaDataGenerator(X_val, y_val, fma_small_dir, batch_size=batch_size, shuffle=False, augment=False)
    
    return train_gen, val_gen, num_classes, encoder

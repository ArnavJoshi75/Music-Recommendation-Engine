import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Reshape, LSTM, Dense, Dropout, Input, SpatialDropout2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import os

def build_crnn_model(input_shape, num_classes):
    """
    Builds a Convolutional Recurrent Neural Network (CRNN) for audio classification.
    
    Args:
        input_shape (tuple): Shape of the input Mel-spectrograms e.g., (time_steps, n_mels, 1)
        num_classes (int): Number of target classes/genres.
    
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # --- CNN Component (Extracts spatial/timbral features from the spectrogram) ---
        
        # Block 1
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.2),
        
        # Block 2
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.2),
        
        # Block 3
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.2),
        
        # Block 4
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D((2, 4)), # More pooling on the frequency axis to flatten it out
        SpatialDropout2D(0.2),
    ])
    
    # Calculate the dynamically pooled shape
    # Output shape after CNN is (None, time_steps', freq', channels')
    shape = model.output_shape
    new_time_steps = shape[1]
    features = shape[2] * shape[3]
    
    # Sequence Formatting
    # Reshape back to a sequence: (batch, time_steps', features)
    model.add(Reshape((new_time_steps, features)))
    
    # --- RNN Component (Captures temporal context and rhythm) ---
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False)) # Summarizes the sequence into a single vector
    
    # --- Classification Head ---
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))
    
    # Compile the model
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def save_model_plot(model, save_dir="visualizations/model"):
    """Saves the model architecture diagram and text summary."""
    os.makedirs(save_dir, exist_ok=True)
    try:
        plot_model(model, to_file=os.path.join(save_dir, 'model_architecture.png'), show_shapes=True, show_layer_names=True)
        print(f"Model architecture diagram saved to {save_dir}")
    except Exception as e:
        print(f"Could not save visual model diagram (Graphviz may be missing): {e}")
        
    # Always save summary as text
    with open(os.path.join(save_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary text saved to {save_dir}")

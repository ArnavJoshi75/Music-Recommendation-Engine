import os
import numpy as np
from data_generator import prepare_data, save_spectrogram_plot
from crnn_model import build_crnn_model, save_model_plot
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def main():
    metadata_path = os.path.join('Dataset', 'fma_metadata')
    fma_small_dir = os.path.join('Dataset', 'fma_small')
    
    # 1. Prepare Data Generators
    print("Initializing Data Pipeline...")
    train_gen, val_gen, num_classes, encoder = prepare_data(
        metadata_path=metadata_path, 
        fma_small_dir=fma_small_dir, 
        batch_size=32
    )
    
    # Check if there's any data
    if len(train_gen) == 0:
        print("Error: No training data generated. Check dataset paths.")
        return

    # 2. Determine Input Shape dynamically from the first batch
    print("Determining input shape...")
    X, y = train_gen[0]
    input_shape = X.shape[1:]  # (time_steps, n_mels, 1)
    print(f"Computed model input shape: {input_shape}")
    
    # Save a sample spectrogram visualization
    print("Saving sample spectrogram...")
    sample_genre_idx = np.argmax(y[0])
    sample_genre_name = encoder.classes_[sample_genre_idx]
    save_spectrogram_plot(X[0], sample_genre_name)
    
    # 3. Build or Load Model
    model_path = 'best_crnn_model.keras'
    if os.path.exists(model_path):
        from tensorflow.keras.models import load_model
        print(f"\nFound existing model at {model_path}!")
        print("Resuming training from saved weights...")
        model = load_model(model_path)
    else:
        print("\nBuilding new CRNN Model...")
        model = build_crnn_model(input_shape, num_classes)
        save_model_plot(model)
        
    model.summary()
    
    # 4. Callbacks for saving best model and stopping early if plateauing
    checkpoint = ModelCheckpoint(
        'best_crnn_model.keras', 
        save_best_only=True, 
        monitor='val_accuracy', 
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
    
    # 5. Train the Model
    print("\n===============================")
    print("      Starting Training        ")
    print("===============================\n")
    
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=30,
            callbacks=[checkpoint, early_stop, reduce_lr]
        )
        print("\nTraining completed successfully!")
        
        # Save training history plots
        print("Saving training history plots...")
        save_training_history(history)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")

def save_training_history(history, save_dir="visualizations/training"):
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    
    # Accuracy Plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()

    # Loss Plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    print(f"Saved training history curves to {save_dir}")

if __name__ == "__main__":
    main()
"""
evaluate_models.py
==================
Evaluates various versions of the CRNN model on the validation set.
Generates classification reports (Precision, Recall, F1) and Confusion Matrices,
saving them to the visualizations/model_reports folder for easy comparison.

Usage:
    python evaluate_models.py
"""

import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import prepare_data

def evaluate_model(model_path, val_gen, encoder, save_dir):
    print(f"\nEvaluating Model: {model_path}")
    print("-" * 50)
    
    # 1. Load Model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return
        
    # 2. Get Predictions
    print("Running predictions on validation set...")
    y_true = []
    y_pred = []
    
    # Iterate through the validation generator
    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        preds = model.predict(X_batch, verbose=0)
        
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
        print(f"Processed batch {i+1}/{len(val_gen)}", end='\r')
    print("\n")
    
    # 3. Generate Classification Report
    genre_names = list(encoder.classes_)
    report = classification_report(y_true, y_pred, target_names=genre_names)
    print(report)
    
    # Save Report
    model_name = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
    report_path = os.path.join(save_dir, f"{model_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report for: {model_name}\n")
        f.write("="*50 + "\n")
        f.write(report)
    print(f"Saved text report to: {report_path}")
    
    # 4. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_names, yticklabels=genre_names)
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Genre', fontsize=12)
    plt.xlabel('Predicted Genre', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

def main():
    metadata_path = os.path.join('Dataset', 'fma_metadata')
    fma_small_dir = os.path.join('Dataset', 'fma_small')
    save_dir = os.path.join('visualizations', 'model_reports')
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Prepare Data
    print("Initializing Data Pipeline to get Validation Set...")
    # Get the validation generator
    _, val_gen, num_classes, encoder = prepare_data(
        metadata_path=metadata_path, 
        fma_small_dir=fma_small_dir, 
        batch_size=32
    )
    
    if len(val_gen) == 0:
        print("Error: Validation generator is empty. Check your dataset paths.")
        return
        
    # 2. Find Models
    models = glob.glob("*.keras") + glob.glob("*.h5")
    if not models:
        print("No .keras or .h5 models found in the current directory.")
        return
        
    print(f"\nFound {len(models)} model(s) to evaluate: {models}")
    
    # 3. Evaluate each model
    for model_path in models:
        evaluate_model(model_path, val_gen, encoder, save_dir)
        
    print(f"\nAll models evaluated. Reports and Confusion Matrices saved in '{save_dir}/'")

if __name__ == "__main__":
    main()

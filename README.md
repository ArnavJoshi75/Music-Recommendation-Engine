# CRNN Music Genre Classification & Recommendation Engine

This repository contains an end-to-end Deep Learning pipeline for classifying music genres and building a content-based recommendation engine. It processes raw audio waveforms into Mel-Spectrograms and uses a hybrid **Convolutional Recurrent Neural Network (CRNN)** to extract robust acoustic features.

## 🚀 Features

- **Dynamic Data Pipeline:** Uses a custom `tf.keras.utils.Sequence` generator to load and process audio on-the-fly, preventing memory overflow.
- **Advanced Augmentations:** Implements SpecAugment (time and frequency masking) and Gaussian noise injections to prevent model overfitting.
- **Hybrid CRNN Architecture:** Combines 4 Convolutional blocks (spatial pattern extraction) with an LSTM sequence model (temporal rhythm tracking).
- **Content-Based Recommendation Engine:** Bypasses the classification head to extract 64-dimensional latent embeddings, enabling nearest-neighbor audio similarity search.
- **Vector Space Visualization:** Built-in t-SNE dimensionality reduction to visually map the acoustic clustering of genres.

---

## 📁 Repository Structure

```text
├── crnn_model.py                     # Defines the CRNN Keras architecture
├── data_generator.py                 # Custom Data Loader, Preprocessing, and Augmentations
├── train.py                          # Training loop with Callbacks (EarlyStopping, ReduceLROnPlateau)
├── predict.py                        # Standalone script for running inference on new audio files
├── evaluate_models.py                # Generates Classification Reports and Confusion Matrices
│
└── recommendation_engine/            # Content-Based Audio Retrieval Module
    ├── feature_extractor.py          # Modifies the CRNN into a "headless" embedding extractor
    ├── build_vector_db.py            # Converts audio datasets into a vector database (.npy)
    ├── recommend.py                  # Nearest-Neighbors script for finding similar songs
    └── visualize_vector_space.py     # Maps 64D embeddings into a 2D t-SNE scatter plot
```

*(Note: Datasets, trained model weights `.keras`, and the virtual environment are excluded from this repository.)*

---

## 💿 Dataset

This project is configured to train on the **Free Music Archive (FMA) Small** subset:
- **Tracks:** 8,000 (30 seconds each)
- **Genres:** 8 (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock) - Perfectly balanced at 1,000 tracks per genre.

To train the model yourself, you must download the FMA metadata and audio files into a folder named `Dataset/` in the root directory.

---

## 🛠 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ArnavJoshi75/Music-Recommendation-Engine.git
   cd Music-Recommendation-Engine
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install required dependencies:**
   *(Ensure you install the equivalent packages: TensorFlow, librosa, pandas, numpy, scikit-learn, matplotlib, seaborn)*
   ```bash
   pip install tensorflow librosa pandas numpy scikit-learn matplotlib seaborn
   ```

---

## 💻 Usage

### 1. Training the Model
Train the CRNN from scratch. The script uses early stopping and learning rate plateaus for optimal convergence.
```bash
python train.py
```

### 2. Evaluating Models
Generate a detailed classification report (Precision, Recall, F1) and a Confusion Matrix heat map for any `.keras` models in the root folder.
```bash
python evaluate_models.py
```

### 3. Running Inference
Predict the genre of a single, unseen audio file:
```bash
python predict.py "path/to/your/song.mp3"
```

### 4. Building the Recommendation Engine
Convert all tracks into acoustic embeddings and query for similar songs:
```bash
# Step 1: Build the Vector Database (processes all tracks)
python recommendation_engine/build_vector_db.py

# Step 2: Get Recommendations for a specific track
python recommendation_engine/recommend.py "path/to/query/song.mp3"

# Step 3: Visualize the Vector Space
python recommendation_engine/visualize_vector_space.py
```

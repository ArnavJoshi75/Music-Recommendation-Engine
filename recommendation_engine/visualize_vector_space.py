"""
visualize_vector_space.py
=========================
Visualizes the 64-dimensional CRNN embeddings using t-SNE.
Colors the points by their ground-truth music genre to observe
how well the model clusters similar music together.

Usage:
    python recommendation_engine/visualize_vector_space.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Ensure the project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_generator import load_fma_metadata

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(ENGINE_DIR, "embeddings.npy")
TRACK_IDS_PATH = os.path.join(ENGINE_DIR, "track_ids.npy")
METADATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "fma_metadata")
FMA_SMALL_DIR = os.path.join(PROJECT_ROOT, "Dataset", "fma_small")
VISUALIZATIONS_DIR = os.path.join(ENGINE_DIR, "visualizations")


def main():
    print("=" * 60)
    print("   Vector Space Visualization (t-SNE)")
    print("=" * 60)

    # 1. Load the database
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(TRACK_IDS_PATH):
        print("ERROR: Database files not found. Run build_vector_db.py first.")
        sys.exit(1)

    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    track_ids = np.load(TRACK_IDS_PATH)
    print(f"Loaded {len(embeddings)} embeddings of shape {embeddings.shape[1]}-D.")

    # 2. Get the true genres
    print("\nLoading FMA metadata to get ground-truth genres...")
    # This reads tracks.csv and maps tracks to labels
    valid_ids, valid_labels = load_fma_metadata(METADATA_PATH, FMA_SMALL_DIR)
    
    # Create a mapping from track_id to genre
    id_to_genre = dict(zip(valid_ids, valid_labels))
    
    # Map our database track_ids array to genres
    genres = []
    for tid in track_ids:
        genres.append(id_to_genre.get(tid, "Unknown"))
        
    genres = np.array(genres)
    unique_genres = np.unique(genres)
    
    # If a track somehow didn't map, it'll show as Unknown, but normally there shouldn't be any.
    print(f"Found {len(unique_genres)} unique genres: {list(unique_genres)}")

    # 3. Dimensionality Reduction (t-SNE)
    print("\nRunning t-SNE dimensionality reduction (this may take a minute)...")
    # Using PCA initialization is standard practice to improve global structure
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings)

    # 4. Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    
    # Use tab10 colormap which is good for up to 10 distinct categories
    colormap = plt.colormaps.get_cmap('tab10')
    
    for idx, genre in enumerate(unique_genres):
        mask = (genres == genre)
        
        # Color index
        c_idx = idx % 10
        color = colormap(c_idx)
        
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1], 
            label=genre, 
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5,
            c=[color]
        )
        
    plt.title("CRNN Embeddings Vector Space (t-SNE Projection)", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Music Genre", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 5. Save the plot
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    save_path = os.path.join(VISUALIZATIONS_DIR, "vector_space_tsne.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    
    # Also show it directly if in an interactive environment
    plt.show()

if __name__ == "__main__":
    main()

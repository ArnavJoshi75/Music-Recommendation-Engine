"""
recommend.py
============
Content-based music recommendation engine.

Loads the pre-built vector database (embeddings.npy, track_ids.npy),
fits a cosine-similarity KNN model, extracts the embedding of a query
audio file, and returns the top-N most similar tracks from the database.

Also visualizes the Mel spectrograms of the query track and each
recommended track side-by-side for visual comparison.

Usage:
    python recommendation_engine/recommend.py <path_to_query.mp3>
    python recommendation_engine/recommend.py <path_to_query.mp3> --n 10
    python recommendation_engine/recommend.py <path_to_query.mp3> --no-plot

Dependencies:
    - NumPy, Scikit-Learn, TensorFlow, Librosa, Matplotlib
    - Pre-built embeddings.npy and track_ids.npy (run build_vector_db.py first)
"""

import os
import sys
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Ensure the project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from recommendation_engine.feature_extractor import (
    load_headless_model,
    extract_features,
    SAMPLE_RATE,
    HOP_LENGTH,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "best_crnn_model.keras")
EMBEDDINGS_PATH = os.path.join(ENGINE_DIR, "embeddings.npy")
TRACK_IDS_PATH = os.path.join(ENGINE_DIR, "track_ids.npy")
FMA_SMALL_DIR = os.path.join(PROJECT_ROOT, "Dataset", "fma_small")
VISUALIZATIONS_DIR = os.path.join(ENGINE_DIR, "visualizations")

DEFAULT_N_RECOMMENDATIONS = 5


def load_vector_database() -> tuple:
    """
    Loads the pre-computed embedding database from disk.

    Returns:
        (embeddings, track_ids):
            embeddings — np.ndarray of shape (N, 64), float32
            track_ids  — np.ndarray of shape (N,),    int32
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_PATH}\n"
            f"Run build_vector_db.py first to generate the database."
        )
    if not os.path.exists(TRACK_IDS_PATH):
        raise FileNotFoundError(
            f"Track IDs file not found: {TRACK_IDS_PATH}\n"
            f"Run build_vector_db.py first to generate the database."
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    track_ids = np.load(TRACK_IDS_PATH)

    print(f"[VectorDB] Loaded {len(track_ids)} embeddings, "
          f"shape {embeddings.shape}")

    return embeddings, track_ids


def build_knn_index(embeddings: np.ndarray,
                    n_neighbors: int) -> NearestNeighbors:
    """
    Fits a brute-force cosine-similarity KNN model on the embeddings.

    Args:
        embeddings:   (N, 64) float32 array of track embeddings.
        n_neighbors:  Number of nearest neighbors to retrieve.

    Returns:
        Fitted NearestNeighbors instance.
    """
    # We request n_neighbors+1 because the query track itself may be
    # present in the database (distance ≈ 0). We filter it out later.
    knn = NearestNeighbors(
        n_neighbors=min(n_neighbors + 1, len(embeddings)),
        metric="cosine",
        algorithm="brute",
    )
    knn.fit(embeddings)
    print(f"[KNN] Index fitted with metric='cosine', "
          f"n_neighbors={knn.n_neighbors}")
    return knn


def get_query_embedding(audio_path: str, model) -> np.ndarray:
    """
    Extracts the 64-D embedding of a query audio file.

    Args:
        audio_path:  Path to the query .mp3 file.
        model:       The headless Keras feature extractor.

    Returns:
        1-D numpy array of shape (64,).
    """
    print(f"[Query] Extracting features from: {audio_path}")
    spectrogram = extract_features(audio_path)      # (1, 1292, 128, 1)
    embedding = model.predict(spectrogram, verbose=0)  # (1, 64)
    return embedding[0]                                # (64,)


def get_fma_track_path(track_id: int) -> str:
    """
    Maps a numeric FMA track ID to its on-disk file path.

    Example: track_id=2 -> Dataset/fma_small/000/000002.mp3
    """
    tid_str = f"{track_id:06d}"
    return os.path.join(FMA_SMALL_DIR, tid_str[:3], tid_str + ".mp3")


def compute_mel_spectrogram_db(audio_path: str) -> np.ndarray:
    """
    Loads an audio file and returns the Mel spectrogram in dB scale,
    shaped (n_mels, time_steps) — ready for librosa.display.specshow().

    Uses the same parameters as the training/feature extraction pipeline.
    """
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=30)
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=2048, hop_length=HOP_LENGTH, n_mels=128
    )
    return librosa.power_to_db(S, ref=np.max)


def _save_single_spectrogram(S_dB: np.ndarray, title: str,
                              save_path: str):
    """
    Renders and saves a single Mel spectrogram to disk.

    Args:
        S_dB:      Mel spectrogram in dB, shape (n_mels, time_steps).
        title:     Title text for the plot.
        save_path: Absolute path for the output PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_dB, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_run_folder(query_label: str) -> str:
    """
    Creates a timestamped subfolder inside visualizations/ for this
    recommendation run.

    Example: visualizations/query_my_song_20260706_000823/

    Returns:
        Absolute path to the created subfolder.
    """
    from datetime import datetime

    # Sanitize the query name for use as a folder name
    safe_name = os.path.splitext(query_label)[0]
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_"
                        for c in safe_name).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"query_{safe_name}_{timestamp}"

    run_dir = os.path.join(VISUALIZATIONS_DIR, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def plot_spectrograms(query_path: str, query_label: str,
                      rec_paths: list, rec_labels: list,
                      save: bool = True):
    """
    Plots the Mel spectrograms of the query track and each recommended
    track in a single comparison figure, and also saves each spectrogram
    as its own individual PNG inside a per-run subfolder.

    Folder structure (when save=True):
        recommendation_engine/visualizations/
          query_<name>_<timestamp>/
            query_<name>.png               ← individual query spectrogram
            rec_1_track_<id>.png           ← individual recommendation
            rec_2_track_<id>.png
            ...
            comparison.png                 ← side-by-side overview

    Args:
        query_path:  Path to the query audio file.
        query_label: Display label for the query track.
        rec_paths:   List of paths to recommended audio files.
        rec_labels:  List of display labels for recommended tracks.
        save:        If True, saves individual + comparison figures.
    """
    # Create a dedicated subfolder for this run
    run_dir = _make_run_folder(query_label) if save else None

    # --- Individual spectrogram: query track ---
    S_query = compute_mel_spectrogram_db(query_path)
    if save:
        safe_query = os.path.splitext(query_label)[0]
        safe_query = "".join(c if c.isalnum() or c in ("-", "_") else "_"
                             for c in safe_query).strip("_")
        query_save = os.path.join(run_dir, f"query_{safe_query}.png")
        _save_single_spectrogram(S_query, f"QUERY — {query_label}", query_save)
        print(f"  Saved: {query_save}")

    # --- Individual spectrograms: each recommendation ---
    rec_spectrograms = []    # store (S_dB, label) for the combined figure
    for i, (path, label) in enumerate(zip(rec_paths, rec_labels), start=1):
        try:
            S_rec = compute_mel_spectrogram_db(path)
            rec_spectrograms.append((S_rec, label))

            if save:
                # Extract track ID from the label for the filename
                tid_part = label.split("\n")[0].replace(" ", "_").lower()
                rec_save = os.path.join(run_dir, f"rec_{i}_{tid_part}.png")
                _save_single_spectrogram(S_rec, f"#{i}  {label}", rec_save)
                print(f"  Saved: {rec_save}")
        except Exception as e:
            rec_spectrograms.append((None, label))
            print(f"  [WARN] Could not process spectrogram for {path}: {e}")

    # ------------------------------------------------------------------
    # Combined comparison figure (query + all recommendations)
    # ------------------------------------------------------------------
    n_total = 1 + len(rec_spectrograms)
    fig, axes = plt.subplots(1, n_total, figsize=(5 * n_total, 4),
                             constrained_layout=True)

    if n_total == 1:
        axes = [axes]

    # Plot query
    img = librosa.display.specshow(
        S_query, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=axes[0], cmap="magma"
    )
    axes[0].set_title(f"QUERY\n{query_label}", fontsize=10, fontweight="bold",
                      color="#00e676")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")

    # Plot recommendations
    for i, (S_rec, label) in enumerate(rec_spectrograms, start=1):
        if S_rec is not None:
            librosa.display.specshow(
                S_rec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                x_axis="time", y_axis="mel", ax=axes[i], cmap="magma"
            )
            axes[i].set_title(f"#{i}  {label}", fontsize=10)
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("")
        else:
            axes[i].set_title(f"#{i}  {label}\n(load error)", fontsize=10,
                              color="red")

    fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.8, pad=0.02)
    fig.suptitle("Mel Spectrograms — Query vs. Recommendations",
                 fontsize=14, fontweight="bold", y=1.02)

    if save:
        comparison_path = os.path.join(run_dir, "comparison.png")
        fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {comparison_path}")
        print(f"\n[Visualization] All spectrograms saved to:\n  {run_dir}")

    plt.show()
    plt.close(fig)


def recommend(audio_path: str, n: int = DEFAULT_N_RECOMMENDATIONS,
              show_plot: bool = True):
    """
    Full recommendation pipeline:
        query.mp3 -> embedding -> KNN search -> top-N similar tracks
        -> visualize Mel spectrograms side-by-side.

    Args:
        audio_path: Path to the query audio file.
        n:          Number of recommendations to return.
        show_plot:  If True, display & save spectrogram comparison.
    """
    # Validate query file
    if not os.path.exists(audio_path):
        print(f"ERROR: Query file not found: {audio_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load the vector database
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("   Music Recommendation Engine")
    print("=" * 60)

    embeddings, track_ids = load_vector_database()

    # ------------------------------------------------------------------
    # 2. Build the KNN index
    # ------------------------------------------------------------------
    knn = build_knn_index(embeddings, n)

    # ------------------------------------------------------------------
    # 3. Load headless model and extract query embedding
    # ------------------------------------------------------------------
    model = load_headless_model(MODEL_PATH)
    query_embedding = get_query_embedding(audio_path, model)

    # ------------------------------------------------------------------
    # 4. KNN search — find nearest neighbors
    # ------------------------------------------------------------------
    distances, indices = knn.kneighbors(
        query_embedding.reshape(1, -1)       # (1, 64) for sklearn
    )

    # Flatten results from the single-query batch
    distances = distances[0]   # cosine distances (0 = identical)
    indices = indices[0]

    # ------------------------------------------------------------------
    # 5. Format and display results
    # ------------------------------------------------------------------
    # Convert cosine distance to cosine similarity (1 = identical)
    similarities = 1.0 - distances

    print("\n" + "=" * 60)
    print(f"   TOP {n} RECOMMENDATIONS")
    print(f"   Query: {os.path.basename(audio_path)}")
    print("=" * 60)
    print(f"  {'Rank':<6} {'Track ID':<12} {'Similarity':<14} {'FMA Path'}")
    print("-" * 60)

    # Collect recommended track info for spectrogram plotting
    rec_track_paths = []
    rec_track_labels = []

    count = 0
    for dist, sim, idx in zip(distances, similarities, indices):
        matched_tid = track_ids[idx]

        # Skip self-matches (cosine distance very close to 0)
        if sim > 0.9999:
            continue

        count += 1
        tid_str = f"{matched_tid:06d}"
        fma_path = f"fma_small/{tid_str[:3]}/{tid_str}.mp3"
        print(f"  #{count:<5} {matched_tid:<12} {sim:<14.4f} {fma_path}")

        # Store path and label for spectrogram visualization
        rec_track_paths.append(get_fma_track_path(matched_tid))
        rec_track_labels.append(f"Track {matched_tid}\n(sim: {sim:.4f})")

        if count >= n:
            break

    # Edge case: if we filtered out the self-match and have fewer results
    if count == 0:
        print("  No recommendations found. The database may be too small.")

    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 6. Visualize Mel spectrograms — query vs. recommendations
    # ------------------------------------------------------------------
    if show_plot and count > 0:
        print("[Visualization] Generating Mel spectrogram comparison...")
        query_label = os.path.basename(audio_path)
        plot_spectrograms(
            query_path=audio_path,
            query_label=query_label,
            rec_paths=rec_track_paths,
            rec_labels=rec_track_labels,
            save=True,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Content-based music recommendation using CRNN embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python recommendation_engine/recommend.py my_song.mp3\n"
            "  python recommendation_engine/recommend.py my_song.mp3 --n 10\n"
        ),
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the query audio file (.mp3, .wav, etc.).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N_RECOMMENDATIONS,
        help=f"Number of recommendations to return (default: {DEFAULT_N_RECOMMENDATIONS}).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Skip spectrogram visualization.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recommend(args.audio_path, n=args.n, show_plot=not args.no_plot)

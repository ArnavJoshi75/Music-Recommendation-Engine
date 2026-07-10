"""
build_vector_db.py
==================
Batch embedding extraction pipeline for the FMA-Small dataset.

Iterates over all ~8,000 validated MP3 tracks, extracts 64-dimensional
latent embeddings from the headless CRNN model, and persists them as
NumPy arrays for downstream similarity search.

Outputs (saved to recommendation_engine/):
    embeddings.npy  — shape (N, 64), float32
    track_ids.npy   — shape (N,),    int

Usage:
    python recommendation_engine/build_vector_db.py

Memory Management:
    - Spectrograms are built one-at-a-time within each batch.
    - After model.predict(), the spectrogram batch is discarded.
    - gc.collect() is called after every batch to reclaim memory.
    - Peak RAM ≈ batch_size × ~650 KB ≈ ~20 MB for batch_size=32.
"""

import os
import sys
import gc
import time
import numpy as np

# Ensure the project root is on sys.path so we can import siblings
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_generator import load_fma_metadata
from recommendation_engine.feature_extractor import (
    load_headless_model,
    extract_features,
    SAMPLE_RATE,
    DURATION,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
)

# Try to import tqdm for progress bars; fall back to a simple counter
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 32          # Number of tracks to process before flushing
MODEL_PATH = os.path.join(PROJECT_ROOT, "best_crnn_model.keras")
METADATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "fma_metadata")
FMA_SMALL_DIR = os.path.join(PROJECT_ROOT, "Dataset", "fma_small")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # recommendation_engine/


def get_audio_path(track_id: int) -> str:
    """
    Maps a numeric FMA track ID to its on-disk path.
    Mirrors the convention in data_generator.FmaDataGenerator._get_audio_path().

    Example: track_id=2 -> Dataset/fma_small/000/000002.mp3
    """
    tid_str = f"{track_id:06d}"
    return os.path.join(FMA_SMALL_DIR, tid_str[:3], tid_str + ".mp3")


def build_spectrogram_batch(audio_paths: list) -> tuple:
    """
    Loads a list of audio files and converts them to Mel-spectrograms.
    Skips files that fail to load (corrupted, missing codec, etc.).

    Args:
        audio_paths: List of (track_id, file_path) tuples.

    Returns:
        (spectrograms, valid_ids):
            spectrograms — np.ndarray of shape (B, 1292, 128, 1)
            valid_ids    — list of track_ids that loaded successfully
    """
    specs = []
    valid_ids = []

    for track_id, path in audio_paths:
        try:
            spec = extract_features(path)       # shape (1, 1292, 128, 1)
            specs.append(spec[0])                # remove batch dim -> (1292, 128, 1)
            valid_ids.append(track_id)
        except Exception as e:
            print(f"  [SKIP] Track {track_id} ({path}): {e}")

    if not specs:
        return np.array([]), valid_ids

    return np.array(specs, dtype=np.float32), valid_ids


def main():
    print("=" * 60)
    print("   FMA-Small Vector Database Builder")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load metadata — get list of valid (track_id, file_path) pairs
    # ------------------------------------------------------------------
    print("\n[Step 1/3] Loading FMA metadata and validating file paths...")
    track_ids, _ = load_fma_metadata(METADATA_PATH, FMA_SMALL_DIR)
    total_tracks = len(track_ids)
    print(f"  Found {total_tracks} valid tracks.\n")

    if total_tracks == 0:
        print("ERROR: No valid tracks found. Check dataset paths.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load the headless feature extraction model
    # ------------------------------------------------------------------
    print("[Step 2/3] Loading headless CRNN model...")
    model = load_headless_model(MODEL_PATH)
    print()

    # ------------------------------------------------------------------
    # 3. Process all tracks in batches
    # ------------------------------------------------------------------
    print(f"[Step 3/3] Extracting embeddings (batch_size={BATCH_SIZE})...")
    start_time = time.time()

    all_embeddings = []   # list of (B, 64) arrays
    all_track_ids = []    # list of track_id ints
    failed_count = 0
    num_batches = (total_tracks + BATCH_SIZE - 1) // BATCH_SIZE

    # Build (track_id, file_path) pairs
    id_path_pairs = [(tid, get_audio_path(tid)) for tid in track_ids]

    # Iterate in batches
    if HAS_TQDM:
        batch_iter = tqdm(range(num_batches), desc="  Batches", unit="batch")
    else:
        batch_iter = range(num_batches)

    for batch_idx in batch_iter:
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total_tracks)
        batch_pairs = id_path_pairs[batch_start:batch_end]

        # Build spectrogram batch
        spec_batch, valid_ids = build_spectrogram_batch(batch_pairs)

        if len(valid_ids) == 0:
            failed_count += (batch_end - batch_start)
            continue

        # Extract embeddings via the headless model
        embeddings = model.predict(spec_batch, verbose=0)   # (B, 64)

        all_embeddings.append(embeddings)
        all_track_ids.extend(valid_ids)
        failed_count += (batch_end - batch_start) - len(valid_ids)

        # Free memory from this batch
        del spec_batch, embeddings
        gc.collect()

        # Fallback progress reporting when tqdm is not installed
        if not HAS_TQDM and (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {batch_end}/{total_tracks} tracks "
                  f"({elapsed:.1f}s elapsed)")

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    elapsed_total = time.time() - start_time

    if not all_embeddings:
        print("\nERROR: No embeddings were extracted. Exiting.")
        sys.exit(1)

    # Concatenate all batch results into single arrays
    final_embeddings = np.concatenate(all_embeddings, axis=0)   # (N, 64)
    final_track_ids = np.array(all_track_ids, dtype=np.int32)   # (N,)

    emb_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
    tid_path = os.path.join(OUTPUT_DIR, "track_ids.npy")

    np.save(emb_path, final_embeddings)
    np.save(tid_path, final_track_ids)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("   BUILD COMPLETE")
    print("=" * 60)
    print(f"  Tracks processed : {len(final_track_ids)}")
    print(f"  Tracks failed    : {failed_count}")
    print(f"  Embedding shape  : {final_embeddings.shape}")
    print(f"  Time elapsed     : {elapsed_total:.1f}s")
    print(f"\n  Saved:")
    print(f"    {emb_path}")
    print(f"    {tid_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

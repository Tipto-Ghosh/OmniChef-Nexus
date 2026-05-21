import os
import json
import torch
import numpy as np
from safetensors.torch import load_file
from datasets import load_dataset, Dataset
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, UpdateStatus
from tqdm import tqdm

from ingestion.config import (
    HF_DATASET_ID,
    DATASET_SPLIT,
    SAFETENSORS_PATH,
    EMBEDDING_KEY,
    COLLECTION_NAME,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    FORCE_INDEX_THRESHOLD,
)
from ingestion.image_utils import pil_to_base64


# Dataset & embeddings loading
def load_hf_dataset() -> Dataset:
    """Load the HuggingFace dataset defined in config.

    Returns:
        Loaded HF ``Dataset`` object.
    """
    print(f"Loading dataset '{HF_DATASET_ID}' (split='{DATASET_SPLIT}') …")
    dataset = load_dataset(HF_DATASET_ID, split=DATASET_SPLIT)
    print(f"  Rows : {len(dataset)}")
    print(f"  Cols : {dataset.column_names}")
    return dataset


def load_embeddings() -> np.ndarray:
    """Load the safetensors file and return embeddings as float32 numpy array.

    Returns:
        Array of shape ``(N, VECTOR_DIM)`` in float32.
    """
    print(f"\nLoading embeddings from '{SAFETENSORS_PATH}' …")
    tensors = load_file(SAFETENSORS_PATH)
    embeddings = tensors[EMBEDDING_KEY].to(torch.float32).numpy()
    print(f"  Shape : {embeddings.shape}")
    print(f"  Dtype : {embeddings.dtype}")
    print(f"  Memory: {embeddings.nbytes / 1024 ** 2:.1f} MB")
    return embeddings


# Checkpoint helpers
def load_checkpoint() -> int:
    """Return the next row index to upsert (0 for a fresh run).

    Reads the JSON checkpoint file written by :func:`save_checkpoint`.

    Returns:
        Index of the first row that still needs upserting.
    """
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        last = data.get("last_upserted_index", -1)
        print(f"[RESUME] Checkpoint found — resuming from index {last + 1}")
        return last + 1
    print("[FRESH]  No checkpoint found — starting from index 0")
    return 0


def save_checkpoint(last_index: int) -> None:
    """Persist the index of the last successfully upserted row.

    Args:
        last_index: The dataset row index of the most recently upserted record.
    """
    os.makedirs(os.path.dirname(CHECKPOINT_PATH) or ".", exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"last_upserted_index": last_index}, f)


# Validation
def validate_embedding(vec: np.ndarray, idx: int) -> bool:
    """Return ``True`` if the embedding vector is finite, ``False`` otherwise.

    Args:
        vec: 1-D float32 numpy array.
        idx: Dataset row index (used in the warning message).

    Returns:
        ``False`` if any element is NaN or Inf; ``True`` otherwise.
    """
    if np.isnan(vec).any() or np.isinf(vec).any():
        print(f"  [WARN] Skipping index {idx} — NaN/Inf in embedding")
        return False
    return True


# Main upsert loop
def run_upsert(
    client: QdrantClient,
    dataset: Dataset,
    embeddings_np: np.ndarray,
) -> dict:
    """Upsert all records into Qdrant with batching and checkpoint support.

    The payload stored per point is:
      - ``"markdown"``  — the markdown recipe text
      - ``"image"``     — base64-encoded JPEG string (or ``None`` on failure)

    Args:
        client:        Connected ``QdrantClient``.
        dataset:       HF dataset with columns ``["markdown", "image", …]``.
        embeddings_np: Float32 numpy array ``(N, VECTOR_DIM)``.

    Returns:
        Summary dict with keys ``total``, ``upserted``, ``skipped``,
        ``failed_images``.
    """
    total_rows    = len(dataset)
    start_index   = load_checkpoint()

    if start_index >= total_rows:
        print("All rows already upserted — nothing to do.")
        return {"total": total_rows, "upserted": 0, "skipped": 0, "failed_images": 0}

    batch: list[PointStruct] = []
    last_upserted  = start_index - 1
    skipped        = 0
    failed_images  = 0

    pbar = tqdm(
        total=total_rows - start_index,
        desc="Upserting",
        unit="recipe",
        dynamic_ncols=True,
    )

    for global_index, row in enumerate(dataset):
        if global_index < start_index:
            continue

        vec = embeddings_np[global_index]
        if not validate_embedding(vec, global_index):
            skipped += 1
            pbar.update(1)
            continue

        markdown_text = row.get("markdown") or ""
        raw_image     = row.get("image")
        image_b64     = pil_to_base64(raw_image)
        if image_b64 is None:
            failed_images += 1

        batch.append(
            PointStruct(
                id=global_index,
                vector=vec.tolist(),
                payload={"markdown": markdown_text, "image": image_b64},
            )
        )

        if len(batch) == BATCH_SIZE:
            result = client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True,
            )
            assert result.status == UpdateStatus.COMPLETED, (
                f"Upsert failed at batch ending index {global_index}"
            )
            last_upserted = global_index
            save_checkpoint(last_upserted)
            batch.clear()

        pbar.update(1)

    # flush remaining records
    if batch:
        result = client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,
        )
        assert result.status == UpdateStatus.COMPLETED, "Final batch upsert failed"
        last_upserted = global_index  # type: ignore[possibly-undefined]
        save_checkpoint(last_upserted)
        batch.clear()

    pbar.close()
    upserted = last_upserted - start_index + 1

    print(f"\n[DONE] Upserted {upserted} records "
          f"({skipped} skipped, {failed_images} image failures)")

    # trigger HNSW index build in background
    print("Triggering HNSW index optimisation …")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config={"indexing_threshold": FORCE_INDEX_THRESHOLD},
    )
    print("Index build queued (runs in background).")

    return {
        "total":         total_rows,
        "upserted":      upserted,
        "skipped":       skipped,
        "failed_images": failed_images,
    }
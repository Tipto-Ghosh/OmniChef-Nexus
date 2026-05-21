import os

# HuggingFace cache
HF_DATASETS_CACHE: str = "D:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE

# Dataset
HF_DATASET_ID: str = "tiptoghosh/food-recipes-15k"
DATASET_SPLIT: str = "train"

# Pre-computed safetensors embeddings
SAFETENSORS_PATH: str = "data/embedding_tensors/all_recipes_image_text_embeddings.safetensors"
EMBEDDING_KEY: str = "image_text_embeddings"

# Qdrant connection
QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333

# Vector configuration
VECTOR_DIM: int = 2048

# imported lazily inside functions to avoid top-level Qdrant dependency
# when config is imported in non-Qdrant contexts
def _distance():
    from qdrant_client.models import Distance
    return Distance.COSINE

DISTANCE = _distance()

# Collections
COLLECTIONS: dict[str, str] = {
    "image_text_index": "image + text combined embeddings",
    "image_index":      "image-only embeddings",
    "text_index":       "text-only embeddings",
}

# The collection we actually upsert into for this pipeline
COLLECTION_NAME: str = "image_text_index"

# HNSW index configuration
HNSW_M: int = 16
HNSW_EF_CONSTRUCT: int = 200
HNSW_FULL_SCAN_THRESHOLD: int = 10_000

# Optimiser configuration
INDEXING_THRESHOLD: int = 20_000   # normal threshold during build
FORCE_INDEX_THRESHOLD: int = 0     # set to 0 after upsert to force HNSW build

# Upsert loop
BATCH_SIZE: int = 64
CHECKPOINT_PATH: str = "data/upsert_checkpoint.json"

# Image processing
MAX_IMAGE_SIZE: int = 512     # max side length (px) before downscaling
JPEG_QUALITY: int = 85        # JPEG compression quality (1-95)
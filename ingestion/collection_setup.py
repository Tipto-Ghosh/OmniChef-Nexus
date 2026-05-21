from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
)

from ingestion.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    VECTOR_DIM,
    DISTANCE,
    COLLECTIONS,
    HNSW_M,
    HNSW_EF_CONSTRUCT,
    HNSW_FULL_SCAN_THRESHOLD,
    INDEXING_THRESHOLD,
)


def get_qdrant_client() -> QdrantClient:
    """Create and return a QdrantClient using settings from config.

    Returns:
        Connected ``QdrantClient`` instance.
    """
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_collections(client: QdrantClient) -> None:
    """Create all collections defined in ``config.COLLECTIONS``.

    Skips a collection if it already exists — delete it manually first
    if you want a completely fresh start.

    Args:
        client: A connected ``QdrantClient`` instance.
    """
    existing_names = {c.name for c in client.get_collections().collections}

    for col_name, description in COLLECTIONS.items():
        if col_name in existing_names:
            print(f"[SKIP]    '{col_name}' already exists.")
            continue

        client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=DISTANCE,
                on_disk=False,   # keep in RAM for fast queries
            ),
            hnsw_config=HnswConfigDiff(
                m=HNSW_M,
                ef_construct=HNSW_EF_CONSTRUCT,
                full_scan_threshold=HNSW_FULL_SCAN_THRESHOLD,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=INDEXING_THRESHOLD,
            ),
        )
        print(f"[CREATED] '{col_name}' → {description}")


def list_collections(client: QdrantClient) -> None:
    """Print a summary of all collections currently in Qdrant.

    Args:
        client: A connected ``QdrantClient`` instance.
    """
    print(f"\n{'Collection':<25} {'Dim':>6}  {'Distance':<10}  {'Points':>8}")
    print("-" * 58)
    for col in client.get_collections().collections:
        info = client.get_collection(col.name)
        vec_cfg = info.config.params.vectors
        print(
            f"{col.name:<25} {vec_cfg.size:>6}  "
            f"{str(vec_cfg.distance):<10}  {info.points_count:>8}"
        )
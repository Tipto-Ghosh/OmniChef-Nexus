from qdrant_client import QdrantClient
from retrieval.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME


def get_qdrant_client(
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
) -> QdrantClient:
    """Create and return a QdrantClient instance.

    Args:
        host: Qdrant host address.
        port: Qdrant gRPC/REST port.

    Returns:
        Connected ``QdrantClient``.
    """
    return QdrantClient(host=host, port=port)


def get_collection_info(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Print a summary of a specific collection's configuration.

    Args:
        client:          Connected ``QdrantClient``.
        collection_name: Name of the collection to inspect.
    """
    info    = client.get_collection(collection_name)
    vec_cfg = info.config.params.vectors

    print(f"Collection : {collection_name}")
    print(f"Points     : {info.points_count}")
    print(f"Dimension  : {vec_cfg.size}")
    print(f"Distance   : {vec_cfg.distance}")
    print(f"Status     : {info.status}")
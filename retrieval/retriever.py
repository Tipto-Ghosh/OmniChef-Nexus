import io
import base64
import warnings

import torch
from PIL import Image
from qdrant_client import QdrantClient
from transformers import AutoModel

from retrieval.config import COLLECTION_NAME
from retrieval.model_loader import prepare_processor


# Internal utility
def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalise a tensor along its last dimension.

    Qdrant stores embeddings that were produced by the same normalisation,
    so we must normalise query vectors before searching.

    Args:
        x:   Tensor of any shape; normalisation applied on dim=-1.
        eps: Small constant to avoid division by zero.

    Returns:
        L2-normalised tensor of the same shape.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def _decode_image(b64_str: str | None) -> Image.Image | None:
    """Decode a base64 JPEG string from the Qdrant payload into a PIL image.

    Args:
        b64_str: Base64-encoded JPEG string, or ``None``.

    Returns:
        PIL ``Image.Image`` or ``None`` on failure.
    """
    if not b64_str:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64_str)))
    except Exception:
        return None


# Public API
def query_qdrant(
    embedding_model: AutoModel,
    qdrant_client: QdrantClient,
    text_query: str | None = None,
    image_query: Image.Image | None = None,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Encode a query and retrieve the top-k results from Qdrant.

    The modality is inferred from which arguments are provided:
      - ``text_query`` only              → ``"text"`` modality
      - ``image_query`` only             → ``"image"`` modality
      - both ``text_query`` + ``image``  → ``"image_text"`` modality

    Args:
        embedding_model:  Loaded embedding model (from ``model_loader``).
        qdrant_client:    Connected Qdrant client (from ``db_client``).
        text_query:       Natural-language query string, or ``None``.
        image_query:      Query PIL image, or ``None``.
        top_k:            Number of results to return.
        collection_name:  Qdrant collection to search.

    Returns:
        List of result dicts, one per hit, sorted by descending score.
        Each dict has keys: ``rank``, ``index``, ``score``,
        ``markdown``, ``image``.

    Raises:
        ValueError: If neither ``text_query`` nor ``image_query`` is provided.
    """
    # 1. infer modality
    if text_query and image_query:
        modality = "image_text"
    elif text_query:
        modality = "text"
    elif image_query:
        modality = "image"
    else:
        raise ValueError("Provide at least one of text_query or image_query.")

    # 2. configure processor and encode
    _, embedding_model = prepare_processor(modality, embedding_model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.inference_mode():
            if modality == "text":
                query_embedding = embedding_model.encode_queries([text_query])
            elif modality == "image":
                query_embedding = embedding_model.encode_documents(images=[image_query])
            else:  # image_text
                query_embedding = embedding_model.encode_documents(
                    texts=[text_query], images=[image_query]
                )

    # 3. normalise to float32 list
    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = torch.tensor(query_embedding)

    query_vec: list[float] = (
        _l2_normalize(query_embedding).squeeze(0).float().tolist()
    )

    # 4. search Qdrant
    hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    )

    # 5. parse and return
    results: list[dict] = []
    for rank, hit in enumerate(hits.points, start=1):
        payload = hit.payload or {}
        results.append({
            "rank":     rank,
            "index":    hit.id,
            "score":    round(hit.score, 4),
            "markdown": payload.get("markdown", ""),
            "image":    _decode_image(payload.get("image")),
        })

    return results
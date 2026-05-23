import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoProcessor
from PIL import Image

from reranker.config import DOC_TEXT_COLUMN, DOC_IMAGE_COLUMN
from reranker.model_loader import build_rerank_processor

def _detect_modality(text_query: str | None, image_query) -> str:
    """Infer query modality from which inputs are provided.

    Args:
        text_query:  Text query string or ``None``.
        image_query: PIL image or ``None``.

    Returns:
        One of ``"text"``, ``"image"``, or ``"image_text"``.

    Raises:
        ValueError: If both inputs are ``None``.
    """
    has_text  = bool(text_query)
    has_image = image_query is not None

    if has_text and has_image:
        return "image_text"
    if has_text:
        return "text"
    if has_image:
        return "image"
    raise ValueError("Provide at least one of text_query or image_query.")


def _build_examples(
    query_text: str | None,
    query_image: Image.Image | None,
    candidate_indices: list[int],
    dataset: Dataset,
    modality: str,
) -> list[dict]:
    """Build the list of (question, doc_text, doc_image) dicts for the processor.

    The processor expects:
      - ``"question"``  — the user's query (text string)
      - ``"doc_text"``  — the document's text content (markdown recipe)
      - ``"doc_image"`` — the document's image (PIL) or ``""`` if not needed

    Args:
        query_text:         Text query or ``None``.
        query_image:        PIL image or ``None``.
        candidate_indices:  Dataset row indices to score.
        dataset:            HF dataset containing doc text and images.
        modality:           One of ``"text"``, ``"image"``, ``"image_text"``.

    Returns:
        List of dicts ready for ``processor.process_queries_documents_crossencoder()``.
    """
    examples = []
    for idx in candidate_indices:
        row = dataset[idx]

        doc_text  = row.get(DOC_TEXT_COLUMN, "") or ""
        doc_image = row.get(DOC_IMAGE_COLUMN)

        example: dict = {"question": query_text or ""}

        if modality == "text":
            example["doc_text"]  = doc_text
            example["doc_image"] = ""
        elif modality == "image":
            example["doc_text"]  = ""
            example["doc_image"] = doc_image
        else:  # image_text
            example["doc_text"]  = doc_text
            example["doc_image"] = doc_image

        examples.append(example)

    return examples

def rerank_results(
    retrieved_results: list[dict],
    dataset: Dataset,
    rerank_model: AutoModelForSequenceClassification,
    query_text: str | None = None,
    query_image=None,
    rerank_processor: AutoProcessor | None = None,
) -> list[dict]:
    """Rerank a list of retrieval results using the cross-encoder model.

    The modality is inferred automatically from ``query_text`` / ``query_image``.
    If ``rerank_processor`` is not supplied, one is built on the fly for the
    detected modality (adds a small one-time cost).

    Args:
        retrieved_results:  List of result dicts from ``query_qdrant()``.
                            Each must have an ``"index"`` key pointing to the
                            HF dataset row.
        dataset:            HF dataset used to look up document text and images.
        rerank_model:       Loaded cross-encoder reranker model.
        query_text:         Text query string, or ``None``.
        query_image:        Query PIL image, or ``None``.
        rerank_processor:   Pre-built processor.  If ``None``, one is built
                            automatically from the detected modality.

    Returns:
        New list of result dicts sorted by descending reranker score, each
        containing: ``rank``, ``index``, ``score`` (raw logit),
        ``softmax_score`` (normalised probability).

    Raises:
        ValueError: If neither ``query_text`` nor ``query_image`` is provided.
    """
    modality = _detect_modality(query_text, query_image)

    # build processor on-the-fly if caller did not supply one
    if rerank_processor is None:
        rerank_processor = build_rerank_processor(modality)

    device = next(rerank_model.parameters()).device

    candidate_indices = [res["index"] for res in retrieved_results]

    # 1. build cross-encoder input examples
    examples = _build_examples(
        query_text=query_text,
        query_image=query_image,
        candidate_indices=candidate_indices,
        dataset=dataset,
        modality=modality,
    )

    # 2. tokenise / patch
    batch_dict = rerank_processor.process_queries_documents_crossencoder(examples)
    batch_dict = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_dict.items()
    }

    # 3. score
    with torch.inference_mode():
        outputs = rerank_model(**batch_dict, return_dict=True)

    # 4. extract logits and softmax
    logits       = outputs.logits.squeeze(-1).cpu()          # (N,)
    probabilities = torch.softmax(logits, dim=0).tolist()

    # 5. sort and format
    scored = [
        {
            "index": candidate_indices[i],
            "score": logits[i].item(),
            "softmax_score": probabilities[i],
        }
        for i in range(len(candidate_indices))
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)

    return [
        {
            "rank": rank,
            "index": item["index"],
            "score": round(item["score"], 4),
            "softmax_score": round(item["softmax_score"], 6),
        }
        for rank, item in enumerate(scored, start=1)
    ]
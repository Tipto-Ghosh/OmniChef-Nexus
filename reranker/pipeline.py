from datasets import Dataset
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor

from retrieval.retriever import query_qdrant
from reranker.reranker   import rerank_results
from reranker.model_loader import build_rerank_processor


def retrieve_and_rerank(
    embedding_model: AutoModel,
    qdrant_client: QdrantClient,
    rerank_model: AutoModelForSequenceClassification,
    dataset: Dataset,
    query_text: str | None = None,
    query_image = None,
    retrieval_top_k: int = 20,
    final_top_n: int = 5,
    rerank_processor: AutoProcessor | None = None,
    collection_name: str | None = None,
) -> dict:
    """Retrieve candidates then rerank and return the top-N results.

    Args:
        embedding_model:  Loaded bi-encoder embedding model (from
                          ``retrieval.model_loader.load_embedding_model``).
        qdrant_client:    Connected Qdrant client.
        rerank_model:     Loaded cross-encoder reranker model (from
                          ``reranker.model_loader.load_rerank_model``).
        dataset:          HF dataset used to look up document text/images for
                          the reranker.
        query_text:       Text query string, or ``None``.
        query_image:      Query PIL image, or ``None``.
        retrieval_top_k:  Number of candidates fetched from Qdrant before
                          reranking.  Larger values give the reranker more to
                          work with at the cost of more inference time.
                          Default: 20.
        final_top_n:      Number of results to return after reranking.
                          Must be ≤ ``retrieval_top_k``.  Default: 5.
        rerank_processor: Pre-built reranker processor.  Built automatically
                          if ``None`` (small one-time cost per call).
        collection_name:  Qdrant collection to search.  Passed through to
                          ``query_qdrant``; uses that function's default if
                          ``None``.

    Returns:
        A dict with three keys:

        ``"retrieved"``
            Full list of ``retrieval_top_k`` results from Qdrant (pre-rerank),
            each with ``rank``, ``index``, ``score``, ``markdown``, ``image``.

        ``"reranked"``
            Full reranked list (same length as ``retrieved``), each with
            ``rank``, ``index``, ``score`` (logit), ``softmax_score``.

        ``"top_n"``
            The final ``final_top_n`` reranked results, enriched with
            ``markdown`` and ``image`` from the retrieval payload so callers
            have everything they need in one place.

    Raises:
        ValueError: If neither ``query_text`` nor ``query_image`` is provided,
                    or if ``final_top_n > retrieval_top_k``.
    """
    if final_top_n > retrieval_top_k:
        raise ValueError(
            f"final_top_n ({final_top_n}) must be ≤ retrieval_top_k ({retrieval_top_k})."
        )

    # retrieval 
    retrieve_kwargs = dict(
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
        text_query=query_text,
        image_query=query_image,
        top_k=retrieval_top_k,
    )
    if collection_name is not None:
        retrieve_kwargs["collection_name"] = collection_name

    retrieved = query_qdrant(**retrieve_kwargs)
    print(f"[Pipeline] Retrieved {len(retrieved)} candidates from Qdrant.")

    # rerank
    reranked = rerank_results(
        retrieved_results=retrieved,
        dataset=dataset,
        rerank_model=rerank_model,
        query_text=query_text,
        query_image=query_image,
        rerank_processor=rerank_processor,
    )
    print(f"[Pipeline] Reranking complete — returning top {final_top_n}.")

    # enrich top-N with retrieval payloads
    # Build a quick index from dataset-index → retrieval payload
    retrieval_map = {r["index"]: r for r in retrieved}

    top_n = []
    for item in reranked[:final_top_n]:
        payload = retrieval_map.get(item["index"], {})
        top_n.append({
            "rank": item["rank"],
            "index": item["index"],
            "rerank_score":  item["score"],
            "softmax_score": item["softmax_score"],
            "retrieval_score": round(payload.get("score", 0.0), 4),
            "markdown": payload.get("markdown", ""),
            "image": payload.get("image"),
        })

    return {
        "retrieved": retrieved,
        "reranked": reranked,
        "top_n": top_n,
    }
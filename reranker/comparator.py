from tabulate import tabulate

def compare_reranking_results(
    before: list[dict],
    after: list[dict],
    title: str = "RETRIEVAL VS. RERANKING COMPARISON",
) -> None:
    """Print a side-by-side comparison table of results before and after reranking.

    Each row shows:
      - New rank after reranking
      - Dataset index
      - Original cosine-similarity score (from the retriever)
      - New reranker logit score
      - Rank movement (e.g. "3 -> 1")
    Args:
        before: List of result dicts from ``query_qdrant()`` (pre-rerank).
                Each dict must have keys ``"rank"``, ``"index"``, ``"score"``.
        after:  List of result dicts from ``rerank_results()`` (post-rerank).
                Each dict must have keys ``"rank"``, ``"index"``, ``"score"``.
        title:  Heading printed above the table.
    """
    before_map = {item["index"]: item for item in before}

    rows = []
    for item_after in after:
        idx = item_after["index"]
        new_rank  = item_after["rank"]
        new_score = item_after["score"]

        if idx in before_map:
            item_before = before_map[idx]
            old_rank = item_before["rank"]
            old_score = item_before["score"]

            delta = old_rank - new_rank
            if delta > 0:
                shift = f"⬆️  +{delta}"
            elif delta < 0:
                shift = f"⬇️  {delta}"
            else:
                shift = "➖  0"

            rank_change = f"{old_rank} ➔ {new_rank}"
            old_score_str = f"{old_score:.4f}"
        else:
            old_rank = "N/A"
            old_score_str = "N/A"
            rank_change = f"N/A ➔ {new_rank}"
            shift = "🆕 New"

        rows.append([
            new_rank,
            idx,
            old_score_str,
            f"{new_score:.4f}",
            rank_change,
            shift,
        ])

    sep = "=" * 80
    headers = ["New Rank", "Dataset Index", "Sim Score", "Rerank Logit", "Rank Change", "Shift"]

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    print(f"{sep}\n")


def summarise_shift(before: list[dict], after: list[dict]) -> dict:
    """Return a summary dict of how many items moved up, down, or stayed.

    Useful for quick programmatic checks without printing the full table.

    Args:
        before: Pre-rerank result list.
        after:  Post-rerank result list.

    Returns:
        Dict with keys ``"moved_up"``, ``"moved_down"``, ``"unchanged"``,
        ``"new"`` (appeared only in after).
    """
    before_map = {item["index"]: item["rank"] for item in before}
    summary = {"moved_up": 0, "moved_down": 0, "unchanged": 0, "new": 0}

    for item in after:
        idx = item["index"]
        new_rank = item["rank"]
        if idx not in before_map:
            summary["new"] += 1
            continue
        delta = before_map[idx] - new_rank
        if delta > 0:
            summary["moved_up"] += 1
        elif delta < 0:
            summary["moved_down"] += 1
        else:
            summary["unchanged"] += 1

    return summary
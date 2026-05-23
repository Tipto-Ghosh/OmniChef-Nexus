import textwrap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from retrieval.config import FIGURE_HEIGHT,AXES_PER_RESULT_WIDTH,MARKDOWN_SNIPPET_CHARS,MARKDOWN_WRAP_WIDTH,MARKDOWN_FONT_SIZE

def display_results(
    results: list[dict],
    query_label: str = "",
    save_path: str | None = None,
) -> None:
    """Render retrieval results as a grid of image + markdown snippet panels.

    Args:
        results:     List of result dicts returned by ``query_qdrant``.
                     Each dict must have keys:
                     ``rank``, ``score``, ``index``, ``image`` (PIL or None),
                     ``markdown`` (str).
        query_label: Label shown in the figure title (typically the query text).
        save_path:   If provided, the figure is saved to this path instead of
                     (or in addition to) being displayed.  Pass a string like
                     ``"output/result.png"`` to export.
    """
    if not results:
        print("[WARN] display_results received an empty result list.")
        return

    n = len(results)
    fig_width = AXES_PER_RESULT_WIDTH * n

    fig = plt.figure(figsize=(fig_width, FIGURE_HEIGHT))
    fig.suptitle(f'Query: "{query_label}"', fontsize=13, y=1.01)

    gs = gridspec.GridSpec(
        2, n,
        height_ratios=[3, 2],
        hspace=0.4,
        wspace=0.3,
    )

    for i, r in enumerate(results):
        # image panel
        ax_img = fig.add_subplot(gs[0, i])

        pil_img: Image.Image | None = r.get("image")
        if pil_img is not None:
            ax_img.imshow(pil_img)
        else:
            ax_img.text(
                0.5, 0.5, "no image",
                ha="center", va="center",
                transform=ax_img.transAxes,
                color="gray", fontsize=9,
            )

        ax_img.axis("off")

        # Accept either key: pipeline.py uses "rerank_score", retriever uses "score"
        rerank_score    = r.get("rerank_score")
        retrieval_score = r.get("retrieval_score") or r.get("score")

        if rerank_score is not None and retrieval_score is not None:
            # came from pipeline.retrieve_and_rerank — show both scores
            title_str = (
                f"#{r['rank']}  rerank: {rerank_score:.4f}\n"
                f"retrieval: {retrieval_score:.4f}  [id: {r['index']}]"
            )
        elif rerank_score is not None:
            title_str = f"#{r['rank']}  rerank: {rerank_score:.4f}\n[id: {r['index']}]"
        elif retrieval_score is not None:
            title_str = f"#{r['rank']}  score: {retrieval_score:.4f}\n[id: {r['index']}]"
        else:
            title_str = f"#{r['rank']}  [id: {r['index']}]"

        ax_img.set_title(title_str, fontsize=9)

        # markdown snippet panel 
        ax_txt = fig.add_subplot(gs[1, i])
        ax_txt.axis("off")

        snippet = r.get("markdown", "")[:MARKDOWN_SNIPPET_CHARS].strip()
        wrapped = textwrap.fill(snippet, width=MARKDOWN_WRAP_WIDTH)

        ax_txt.text(
            0, 1,
            wrapped,
            fontsize=MARKDOWN_FONT_SIZE,
            va="top",
            ha="left",
            transform=ax_txt.transAxes,
            family="monospace",
            wrap=True,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[SAVED] Figure written to '{save_path}'")

    plt.show()


def display_single(result: dict, query_label: str = "") -> None:
    """Convenience wrapper to visualise a single result dict.

    Args:
        result:      A single result dict (same format as ``display_results``).
        query_label: Label shown in the figure title.
    """
    display_results([result], query_label=query_label)
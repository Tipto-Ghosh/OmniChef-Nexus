"""
pipeline2.py
Main runner for the recipe review card pipeline.
Output: PNG images only (PDF is built in-memory then converted, not saved).
"""

import os
import sys
import ast
import logging
import argparse
import tempfile
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Any

import fitz  # PyMuPDF — pure Python, no Poppler needed

from config import (
    pick_variation, IMAGE_SUBDIR, FILENAME_PAD, PNG_DPI,
)
from utils import (
    parse_name, parse_minutes, parse_rating, parse_tags,
    parse_nutrition, parse_steps, parse_ingredients,
    parse_description, parse_reviews,
)
from builder import build_recipe_card, _font_scale

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc="", unit="it", **kwargs):
        total = total or (len(iterable) if hasattr(iterable, "__len__") else "?")
        for i, item in enumerate(iterable, 1):
            print(f"\r{desc}: {i}/{total} {unit}s", end="", flush=True)
            yield item
        print()


# ══════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline2_{ts}.log"
    fmt      = "%(asctime)s | %(levelname)-8s | %(message)s"

    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt, datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    for noisy in ("matplotlib", "PIL", "reportlab"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger("pipeline2")
    logger.info("Log: %s", log_file)
    return logger


# ══════════════════════════════════════════════════════════════════
#  PDF PAGE COUNTER (pure Python, no poppler)
# ══════════════════════════════════════════════════════════════════

def get_page_count(pdf_path: str) -> int:
    """Return number of pages in a PDF using PyMuPDF (pure Python, no Poppler)."""
    try:
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return n
    except Exception as e:
        logging.warning("Could not read PDF page count: %s", e)
        return 1  # Assume 1 page if unreadable


# ══════════════════════════════════════════════════════════════════
#  DATASET LOADER
# ══════════════════════════════════════════════════════════════════

def _safe_eval(val, fallback=None):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback
    if not isinstance(val, str):
        return val
    try:
        return ast.literal_eval(val.strip())
    except Exception:
        return fallback


def load_dataset(data_path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading: %s", data_path)

    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        try:
            df = pd.read_parquet(data_path)
        except Exception:
            df = pd.read_csv(data_path)

    logger.info("Loaded %d rows | columns: %s", len(df), list(df.columns))

    if "review" in df.columns:
        df["review"] = df["review"].apply(
            lambda v: v if isinstance(v, list)
            else _safe_eval(v, fallback=[])
        )

    return df


# ══════════════════════════════════════════════════════════════════
#  FILENAME
# ══════════════════════════════════════════════════════════════════

def _make_stem(recipe_id: Any) -> str:
    """Return filename stem: recipe_id_63986"""
    try:
        return f"recipe_id_{int(float(recipe_id))}"
    except (TypeError, ValueError):
        return "recipe_id_unknown"


# ══════════════════════════════════════════════════════════════════
#  SINGLE ROW PROCESSOR
# ══════════════════════════════════════════════════════════════════

def process_row(
    row,
    row_index:  int,
    image_dir:  Path,
    logger:     logging.Logger,
    resume:     bool,
) -> dict:
    result = {
        "index":    row_index,
        "name":     "",
        "png":      "",
        "status":   "ok",
        "skipped":  False,
        "error":    "",
    }

    try:
        cfg       = pick_variation(row_index)
        recipe_id = row.get("recipe_id")
        stem      = _make_stem(recipe_id)
        png_path  = image_dir / f"{stem}.png"

        result["png"] = str(png_path)

        if resume and png_path.exists():
            result["skipped"] = True
            return result

        name        = parse_name(row.get("name"))
        cook_time   = parse_minutes(row.get("minutes"))
        rating      = parse_rating(row.get("rating"), row.get("num of ratings"))
        tags        = parse_tags(row.get("tags"))
        nutri       = parse_nutrition(row.get("nutrition"))
        steps       = parse_steps(row.get("steps"))
        desc        = parse_description(row.get("description"))
        ingredients = parse_ingredients(row.get("ingredients"))
        reviews     = parse_reviews(row.get("review"), seed=row_index)
        n_ing       = row.get("n_ingredients")
        n_steps_val = row.get("n_steps")

        result["name"] = name or f"recipe_{row_index+1}"

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_pdf = tmp.name

        # Initial build
        current_max_reviews = len(reviews) if reviews else 0
        current_scale = None
        ok = build_recipe_card(
            name=name, cook_time=cook_time, rating=rating,
            tags=tags, description=desc,
            ingredients=ingredients, steps=steps,
            nutri=nutri, reviews=reviews,
            n_ing=n_ing, n_steps=n_steps_val,
            recipe_id=recipe_id, out_path=tmp_pdf, cfg=cfg,
        )
        if not ok:
            raise RuntimeError("build_recipe_card returned False")

        page_count = get_page_count(tmp_pdf)
        max_attempts = 5
        attempt = 0

        while page_count > 1 and attempt < max_attempts:
            attempt += 1
            logger.debug(
                "Row %d: page count %d > 1, attempt %d",
                row_index + 1, page_count, attempt
            )

            if current_max_reviews > 0:
                current_max_reviews -= 1
                logger.debug("Reducing max reviews to %d", current_max_reviews)
            else:
                if current_scale is None:
                    current_scale = _font_scale(
                        n_steps=len(steps) if steps else 0,
                        n_reviews=len(reviews) if reviews else 0,
                        has_description=bool(desc),
                    )
                current_scale *= 0.9
                if current_scale < 0.6:
                    logger.warning("Font scale below 0.6, stopping retries")
                    break
                logger.debug("Reducing font scale to %.2f", current_scale)

            os.unlink(tmp_pdf)
            ok = build_recipe_card(
                name=name, cook_time=cook_time, rating=rating,
                tags=tags, description=desc,
                ingredients=ingredients, steps=steps,
                nutri=nutri, reviews=reviews,
                n_ing=n_ing, n_steps=n_steps_val,
                recipe_id=recipe_id, out_path=tmp_pdf, cfg=cfg,
                max_reviews=current_max_reviews if current_max_reviews < (len(reviews) if reviews else 0) else None,
                scale_override=current_scale,
            )
            if not ok:
                raise RuntimeError("build_recipe_card returned False on retry")
            page_count = get_page_count(tmp_pdf)

        if page_count > 1:
            logger.warning(
                "Row %d: still %d pages after %d attempts; using first page only",
                row_index + 1, page_count, attempt
            )

        # Convert only first page to PNG (PyMuPDF, no Poppler required)
        doc  = fitz.open(tmp_pdf)
        page = doc[0]
        mat  = fitz.Matrix(PNG_DPI / 72, PNG_DPI / 72)   # 72 pt = 1 inch baseline
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(png_path))
        doc.close()
        os.unlink(tmp_pdf)

        logger.debug(
            "OK    [%04d] %-45s | scheme=%-14s chart=%s",
            row_index + 1, result["name"][:45],
            cfg.scheme.name, cfg.chart,
        )

    except Exception as exc:
        result["status"] = "error"
        result["error"]  = str(exc)
        logger.error("ERROR [%04d] %s\n%s",
                     row_index + 1, result["name"],
                     traceback.format_exc())
        try:
            if 'tmp_pdf' in locals() and os.path.exists(tmp_pdf):
                os.unlink(tmp_pdf)
        except Exception:
            pass

    return result


# ══════════════════════════════════════════════════════════════════
#  SUMMARY & CLI (unchanged)
# ══════════════════════════════════════════════════════════════════

def _summary(results: list, logger: logging.Logger, elapsed: float):
    total   = len(results)
    ok      = sum(1 for r in results if r["status"] == "ok" and not r["skipped"])
    skipped = sum(1 for r in results if r["skipped"])
    errors  = sum(1 for r in results if r["status"] == "error")

    logger.info("=" * 58)
    logger.info("PIPELINE 2 COMPLETE")
    logger.info("=" * 58)
    logger.info("Total rows   : %d", total)
    logger.info("Generated OK : %d", ok)
    logger.info("Skipped      : %d  (--resume)", skipped)
    logger.info("Errors       : %d", errors)
    logger.info("Elapsed      : %.1f s  (%.2f s/row)",
                elapsed, elapsed / max(total, 1))
    logger.info("=" * 58)

    if errors:
        logger.warning("Failed rows:")
        for r in results:
            if r["status"] == "error":
                logger.warning("  [%04d] %s → %s",
                               r["index"]+1, r["name"][:50], r["error"])


def parse_args():
    p = argparse.ArgumentParser(description="Recipe review card PNG pipeline")
    p.add_argument("--data",    required=True,
                   help="Path to dataset (.parquet or .csv)")
    p.add_argument("--output",  required=True,
                   help="Root output directory (images/ created inside)")
    p.add_argument("--limit",   type=int, default=None,
                   help="Process only the first N rows")
    p.add_argument("--resume",  action="store_true",
                   help="Skip rows whose PNG already exists")
    p.add_argument("--log-dir", default=None,
                   help="Log directory (default: <output>/logs)")
    return p.parse_args()


def main():
    args        = parse_args()
    output_root = Path(args.output)
    log_dir     = Path(args.log_dir) if args.log_dir else output_root / "logs"
    logger      = setup_logging(log_dir)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    image_dir = output_root / IMAGE_SUBDIR
    image_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Images → %s", image_dir)
    logger.info("DPI    → %d", PNG_DPI)

    df = load_dataset(data_path, logger)

    if args.limit:
        df = df.head(args.limit)
        logger.info("Limit: %d rows", len(df))

    results = []
    start   = datetime.now()

    for row_index, row in tqdm(df.iterrows(), total=len(df),
                                desc="Generating", unit="card"):
        results.append(process_row(
            row=row, row_index=row_index,
            image_dir=image_dir,
            logger=logger, resume=args.resume,
        ))

    elapsed = (datetime.now() - start).total_seconds()
    _summary(results, logger, elapsed)

    sys.exit(1 if any(r["status"] == "error" for r in results) else 0)


if __name__ == "__main__":
    main()
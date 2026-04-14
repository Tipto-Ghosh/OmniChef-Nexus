"""
utils2.py
Parsers for every column type in the recipe review dataset.

Columns and their raw types:
    name           : str
    minutes        : numpy.int64
    tags           : str  → list of strings
    nutrition      : str  → list of 7 floats
    n_steps        : numpy.int64
    steps          : str  → list of strings
    description    : str
    ingredients    : str  → list of strings
    n_ingredients  : numpy.int64
    recipe_id      : numpy.float64
    rating         : numpy.float64
    review         : list (already parsed) or str (string-encoded list)
    num_of_ratings : numpy.float64

All parsers follow the same contract:
    - Return None / [] / {} if the value is missing, None, NaN, or unparseable
    - Never raise — log a debug warning and return the safe fallback
    - Caller checks the return value and skips the section if falsy
"""

import ast
import re
import html
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from config import (
    NUTRITION_KEYS, NUTRITION_LABELS, NUTRITION_UNITS,
    MAX_REVIEWS, MAX_REVIEW_CHARS,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════

def _is_missing(val: Any) -> bool:
    """True if val is None, NaN float, or empty string."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def _safe_eval_list(val: Any) -> Optional[List]:
    """
    Parse a string-encoded Python list → Python list.
    Returns None if val is missing or unparseable.
    Already-a-list input is returned as-is.
    """
    if _is_missing(val):
        return None
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if not isinstance(val, str):
        return None
    try:
        result = ast.literal_eval(val.strip())
        return list(result) if isinstance(result, (list, tuple)) else None
    except Exception:
        logger.debug("Could not parse list literal: %s", str(val)[:60])
        return None


# ══════════════════════════════════════════════════════════════════
#  NAME
# ══════════════════════════════════════════════════════════════════

def parse_name(val: Any) -> Optional[str]:
    """Return cleaned recipe name, or None if missing."""
    if _is_missing(val):
        return None
    name = str(val).strip().title()
    return name if name else None


# ══════════════════════════════════════════════════════════════════
#  MINUTES
# ══════════════════════════════════════════════════════════════════

def parse_minutes(val: Any) -> Optional[str]:
    """
    Return a human-readable cook time string, or None if missing.
    Clips unreasonably large values (> 10 days) as data artifacts.
    """
    if _is_missing(val):
        return None
    try:
        mins = int(val)
    except (TypeError, ValueError):
        return None
    if mins <= 0:
        return None
    # Clip obvious data errors
    if mins > 14400:   # > 10 days
        logger.debug("Clipping unreasonable minutes value: %d", mins)
        return None
    if mins < 60:
        return f"{mins} min"
    hours, remainder = divmod(mins, 60)
    if remainder == 0:
        return f"{hours} hr"
    return f"{hours} hr {remainder} min"


# ══════════════════════════════════════════════════════════════════
#  RATING
# ══════════════════════════════════════════════════════════════════

def parse_rating(val: Any, num_ratings: Any = None) -> Optional[str]:
    """
    Return formatted rating string with star display, or None if missing.
    e.g.  '4.4 / 5  ★★★★☆  (19 ratings)'
    """
    if _is_missing(val):
        return None
    try:
        rating = float(val)
    except (TypeError, ValueError):
        return None
    if np.isnan(rating) or rating < 0:
        return None

    stars_filled = round(rating)
    stars_filled = max(0, min(5, stars_filled))
    star_str = "★" * stars_filled + "☆" * (5 - stars_filled)

    base = f"{rating:.1f}/5  {star_str}"

    if not _is_missing(num_ratings):
        try:
            n = int(float(num_ratings))
            base += f"  ({n} ratings)"
        except (TypeError, ValueError):
            pass

    return base


# ══════════════════════════════════════════════════════════════════
#  TAGS
# ══════════════════════════════════════════════════════════════════

def parse_tags(val: Any, max_tags: int = 8) -> Optional[List[str]]:
    """
    Return up to max_tags cleaned tag strings, or None if missing.
    """
    tags = _safe_eval_list(val)
    if not tags:
        return None
    cleaned = [str(t).strip().replace("-", " ") for t in tags if str(t).strip()]
    return cleaned[:max_tags] if cleaned else None


# ══════════════════════════════════════════════════════════════════
#  NUTRITION
#  Raw format: string-encoded list of 7 floats
#  [calories, total_fat_pdv, sugar_pdv, sodium_pdv,
#   protein_pdv, sat_fat_pdv, carbs_pdv]
# ══════════════════════════════════════════════════════════════════

def parse_nutrition(val: Any) -> Optional[Dict[str, float]]:
    """
    Parse the nutrition list into a keyed dict.
    Returns None if the list is missing or shorter than expected.
    Any individual NaN value is replaced with 0.0 (partial data shown).
    """
    items = _safe_eval_list(val)
    if not items:
        return None
    if len(items) < len(NUTRITION_KEYS):
        logger.debug(
            "Nutrition list has %d items, expected %d",
            len(items), len(NUTRITION_KEYS),
        )
        return None

    result: Dict[str, float] = {}
    for key, raw in zip(NUTRITION_KEYS, items):
        try:
            fval = float(raw)
            result[key] = 0.0 if np.isnan(fval) else fval
        except (TypeError, ValueError):
            result[key] = 0.0

    return result


# ══════════════════════════════════════════════════════════════════
#  STEPS
# ══════════════════════════════════════════════════════════════════

def parse_steps(val: Any) -> Optional[List[str]]:
    """
    Return a list of numbered instruction steps, or None if missing.
    Each step is capitalised and truncated at 300 chars to avoid overflow.
    """
    items = _safe_eval_list(val)
    if not items:
        return None
    steps = []
    for i, step in enumerate(items, 1):
        text = str(step).strip()
        if not text:
            continue
        text = text[0].upper() + text[1:]   # capitalise first letter
        if len(text) > 300:
            text = text[:297] + "..."
        steps.append(f"{i}. {text}")
    return steps if steps else None


# ══════════════════════════════════════════════════════════════════
#  INGREDIENTS
# ══════════════════════════════════════════════════════════════════

def parse_ingredients(val: Any) -> Optional[List[str]]:
    """
    Return a list of ingredient strings, or None if missing.
    """
    items = _safe_eval_list(val)
    if not items:
        return None
    cleaned = [str(i).strip() for i in items if str(i).strip()]
    return cleaned if cleaned else None


# ══════════════════════════════════════════════════════════════════
#  DESCRIPTION
# ══════════════════════════════════════════════════════════════════

def parse_description(val: Any, max_chars: int = 400) -> Optional[str]:
    """
    Return cleaned description text, truncated if too long.
    """
    if _is_missing(val):
        return None
    text = str(val).strip()
    if not text:
        return None
    # Truncate gracefully at word boundary
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


# ══════════════════════════════════════════════════════════════════
#  REVIEWS
#  Raw type: already a Python list, or string-encoded list.
#
#  Preprocessing steps per review:
#    1. Decode HTML entities  (&amp; → &,  &#39; → ')
#    2. Collapse whitespace / newlines
#    3. Strip leading/trailing quotes and punctuation artifacts
#    4. Truncate at MAX_REVIEW_CHARS with ellipsis
#    5. Skip empty or very short reviews (< 15 chars after cleaning)
#
#  Then sample up to MAX_REVIEWS randomly (seeded for reproducibility).
# ══════════════════════════════════════════════════════════════════

_QUOTE_RE    = re.compile(r'^["\'\s]+|["\'\s]+$')
_WHITESPACE  = re.compile(r'\s+')


def _clean_review(text: str) -> Optional[str]:
    """Clean a single review string. Returns None if too short after cleaning."""
    # HTML entities
    text = html.unescape(text)
    # Collapse whitespace and newlines
    text = _WHITESPACE.sub(" ", text).strip()
    # Strip wrapping quotes/spaces
    text = _QUOTE_RE.sub("", text).strip()
    if len(text) < 15:
        return None
    # Truncate at word boundary
    if len(text) > MAX_REVIEW_CHARS:
        text = text[:MAX_REVIEW_CHARS].rsplit(" ", 1)[0]
        if not text.endswith((".", "!", "?")):
            text += "..."
    return text


def parse_reviews(val: Any, seed: int = 0) -> Optional[List[str]]:
    """
    Return up to MAX_REVIEWS cleaned review strings, or None if missing.
    Reviews are sampled deterministically using the provided seed so the
    same row always gets the same set of reviews across runs.
    """
    items = _safe_eval_list(val)

    # Already a plain list (parquet native type)
    if items is None and isinstance(val, list):
        items = val

    if not items:
        return None

    # Clean each review
    cleaned = []
    for item in items:
        c = _clean_review(str(item))
        if c:
            cleaned.append(c)

    if not cleaned:
        return None

    # Sample deterministically
    if len(cleaned) > MAX_REVIEWS:
        rng = random.Random(seed)
        cleaned = rng.sample(cleaned, MAX_REVIEWS)

    return cleaned


# keep random importable from this module
import random

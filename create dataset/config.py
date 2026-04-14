"""
config2.py
Configuration for the recipe review card pipeline.

Dataset columns:
    name           : str
    minutes        : numpy.int64
    tags           : str  (string-encoded list)
    nutrition      : str  (string-encoded list: [calories, total_fat_pdv, sugar_pdv,
                           sodium_pdv, protein_pdv, sat_fat_pdv, carbs_pdv])
    n_steps        : numpy.int64
    steps          : str  (string-encoded list)
    description    : str
    ingredients    : str  (string-encoded list)
    n_ingredients  : numpy.int64
    recipe_id      : numpy.float64
    rating         : numpy.float64
    review         : list  (already a Python list of review strings)
    num_of_ratings : numpy.float64

Key differences from pipeline 1:
    - No food images
    - Nutrition stored as ordered list (not dict)
    - steps, ingredients, tags stored as string-encoded lists
    - Reviews section added (sample up to 6, preprocess text)
    - Output: PNG only (no PDF saved)
    - One nutrition element per card: chart OR table (not both)
"""

import random
from dataclasses import dataclass
from typing import List
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm


# ══════════════════════════════════════════════════════════════════
#  OUTPUT SETTINGS
# ══════════════════════════════════════════════════════════════════

PNG_DPI        = 300
IMAGE_SUBDIR   = "images"
FILENAME_PAD   = 4           # recipe_0001_...


# ══════════════════════════════════════════════════════════════════
#  PAGE GEOMETRY  (A4)
# ══════════════════════════════════════════════════════════════════

PAGE_W, PAGE_H = A4
MARGIN_H       = 16 * mm
MARGIN_V_TOP   = 12 * mm
MARGIN_V_BOT   = 10 * mm
USABLE_W       = PAGE_W - 2 * MARGIN_H


# ══════════════════════════════════════════════════════════════════
#  NUTRITION COLUMN ORDER
#  The nutrition list has a fixed positional schema:
#  [calories, total_fat_%dv, sugar_%dv, sodium_%dv,
#   protein_%dv, sat_fat_%dv, carbs_%dv]
# ══════════════════════════════════════════════════════════════════

NUTRITION_LABELS = [
    "Calories",
    "Total fat (PDV)",
    "Sugar (PDV)",
    "Sodium (PDV)",
    "Protein (PDV)",
    "Sat. fat (PDV)",
    "Carbs (PDV)",
]
NUTRITION_UNITS = [
    "kcal", "%", "%", "%", "%", "%", "%",
]

# Keys used internally after parsing
NUTRITION_KEYS = [
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "sat_fat_pdv",
    "carbs_pdv",
]

# Macro subset used for charts (skip %DV fields that are less visual)
CHART_KEYS   = ["calories", "total_fat_pdv", "protein_pdv", "carbs_pdv"]
CHART_LABELS = ["Calories", "Fat PDV", "Protein PDV", "Carbs PDV"]


# ══════════════════════════════════════════════════════════════════
#  REVIEW SETTINGS
# ══════════════════════════════════════════════════════════════════

MAX_REVIEWS          = 10     # max number of reviews to show per card
MAX_REVIEW_CHARS     = 200   # truncate individual review text beyond this


# ══════════════════════════════════════════════════════════════════
#  COLOR SCHEMES  (10 themes)
# ══════════════════════════════════════════════════════════════════

@dataclass
class ColorScheme:
    name:            str
    header_bg:       object
    header_text:     object
    accent:          object
    accent2:         object
    badge_text:      object
    body_text:       object
    secondary_text:  object   # tags, meta info
    row_alt:         object
    review_bg:       object   # review card background tint
    chart_colors:    List[str]


COLOR_SCHEMES: List[ColorScheme] = [
    ColorScheme(
        name="WarmSpice",
        header_bg=colors.HexColor("#FDF6EC"),
        header_text=colors.HexColor("#3E2723"),
        accent=colors.HexColor("#E67E22"),
        accent2=colors.HexColor("#C0392B"),
        badge_text=colors.white,
        body_text=colors.HexColor("#3E2723"),
        secondary_text=colors.HexColor("#6D4C41"),
        row_alt=colors.HexColor("#FFF3E0"),
        review_bg=colors.HexColor("#FFF8F0"),
        chart_colors=["#E74C3C","#E67E22","#F1C40F","#2ECC71"],
    ),
    ColorScheme(
        name="FreshHerb",
        header_bg=colors.HexColor("#F1F8E9"),
        header_text=colors.HexColor("#1B5E20"),
        accent=colors.HexColor("#558B2F"),
        accent2=colors.HexColor("#33691E"),
        badge_text=colors.white,
        body_text=colors.HexColor("#212121"),
        secondary_text=colors.HexColor("#388E3C"),
        row_alt=colors.HexColor("#DCEDC8"),
        review_bg=colors.HexColor("#F9FBF5"),
        chart_colors=["#2E7D32","#558B2F","#8BC34A","#CDDC39"],
    ),
    ColorScheme(
        name="OceanBreeze",
        header_bg=colors.HexColor("#E3F2FD"),
        header_text=colors.HexColor("#0D47A1"),
        accent=colors.HexColor("#1565C0"),
        accent2=colors.HexColor("#0277BD"),
        badge_text=colors.white,
        body_text=colors.HexColor("#1A237E"),
        secondary_text=colors.HexColor("#1565C0"),
        row_alt=colors.HexColor("#BBDEFB"),
        review_bg=colors.HexColor("#F0F7FF"),
        chart_colors=["#1565C0","#0288D1","#26C6DA","#4FC3F7"],
    ),
    ColorScheme(
        name="MidnightChef",
        header_bg=colors.HexColor("#212121"),
        header_text=colors.HexColor("#FFD54F"),
        accent=colors.HexColor("#FFB300"),
        accent2=colors.HexColor("#F57F17"),
        badge_text=colors.HexColor("#212121"),
        body_text=colors.HexColor("#212121"),
        secondary_text=colors.HexColor("#555555"),
        row_alt=colors.HexColor("#F5F5F5"),
        review_bg=colors.HexColor("#FAFAFA"),
        chart_colors=["#FFB300","#FF8F00","#F57F17","#FFCA28"],
    ),
    ColorScheme(
        name="BerryFusion",
        header_bg=colors.HexColor("#F3E5F5"),
        header_text=colors.HexColor("#4A148C"),
        accent=colors.HexColor("#7B1FA2"),
        accent2=colors.HexColor("#6A1B9A"),
        badge_text=colors.white,
        body_text=colors.HexColor("#1A237E"),
        secondary_text=colors.HexColor("#7B1FA2"),
        row_alt=colors.HexColor("#E1BEE7"),
        review_bg=colors.HexColor("#FCF5FF"),
        chart_colors=["#8E24AA","#AB47BC","#CE93D8","#F48FB1"],
    ),
    ColorScheme(
        name="SunsetGrill",
        header_bg=colors.HexColor("#FFF8E1"),
        header_text=colors.HexColor("#BF360C"),
        accent=colors.HexColor("#E64A19"),
        accent2=colors.HexColor("#FF6D00"),
        badge_text=colors.white,
        body_text=colors.HexColor("#3E2723"),
        secondary_text=colors.HexColor("#BF360C"),
        row_alt=colors.HexColor("#FFE0B2"),
        review_bg=colors.HexColor("#FFFAF0"),
        chart_colors=["#FF3D00","#FF6D00","#FF9100","#FFCA28"],
    ),
    ColorScheme(
        name="SlateCuisine",
        header_bg=colors.HexColor("#ECEFF1"),
        header_text=colors.HexColor("#263238"),
        accent=colors.HexColor("#455A64"),
        accent2=colors.HexColor("#37474F"),
        badge_text=colors.white,
        body_text=colors.HexColor("#212121"),
        secondary_text=colors.HexColor("#455A64"),
        row_alt=colors.HexColor("#CFD8DC"),
        review_bg=colors.HexColor("#F5F7F8"),
        chart_colors=["#455A64","#546E7A","#78909C","#90A4AE"],
    ),
    ColorScheme(
        name="TropicalBowl",
        header_bg=colors.HexColor("#E0F7FA"),
        header_text=colors.HexColor("#006064"),
        accent=colors.HexColor("#00838F"),
        accent2=colors.HexColor("#00695C"),
        badge_text=colors.white,
        body_text=colors.HexColor("#004D40"),
        secondary_text=colors.HexColor("#00796B"),
        row_alt=colors.HexColor("#B2DFDB"),
        review_bg=colors.HexColor("#F0FFFE"),
        chart_colors=["#00897B","#26A69A","#4DB6AC","#80CBC4"],
    ),
    ColorScheme(
        name="RusticBakery",
        header_bg=colors.HexColor("#EFEBE9"),
        header_text=colors.HexColor("#4E342E"),
        accent=colors.HexColor("#6D4C41"),
        accent2=colors.HexColor("#5D4037"),
        badge_text=colors.white,
        body_text=colors.HexColor("#3E2723"),
        secondary_text=colors.HexColor("#6D4C41"),
        row_alt=colors.HexColor("#D7CCC8"),
        review_bg=colors.HexColor("#FAF7F5"),
        chart_colors=["#6D4C41","#8D6E63","#A1887F","#BCAAA4"],
    ),
    ColorScheme(
        name="CrimsonFeast",
        header_bg=colors.HexColor("#FFEBEE"),
        header_text=colors.HexColor("#B71C1C"),
        accent=colors.HexColor("#C62828"),
        accent2=colors.HexColor("#AD1457"),
        badge_text=colors.white,
        body_text=colors.HexColor("#212121"),
        secondary_text=colors.HexColor("#B71C1C"),
        row_alt=colors.HexColor("#FFCDD2"),
        review_bg=colors.HexColor("#FFF5F5"),
        chart_colors=["#C62828","#D32F2F","#E53935","#EF5350"],
    ),
]


# ══════════════════════════════════════════════════════════════════
#  CHART VARIANTS  (one per card — chart OR table, not both)
# ══════════════════════════════════════════════════════════════════

CHART_VARIANTS: List[str] = [
    "bar",
    "horizontal_bar",
    "radar",
    "table_only",
]


# ══════════════════════════════════════════════════════════════════
#  VARIATION CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass
class VariationConfig:
    scheme: ColorScheme
    chart:  str
    seed:   int


def pick_variation(row_index: int) -> VariationConfig:
    """
    10 schemes × 4 chart types = 40 base combinations.
    Seeded by row_index → reproducible across runs.
    """
    rng = random.Random(row_index)
    return VariationConfig(
        scheme=rng.choice(COLOR_SCHEMES),
        chart=rng.choice(CHART_VARIANTS),
        seed=row_index,
    )

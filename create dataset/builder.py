"""
builder2.py
Single-page recipe review card renderer.

Layout (top to bottom):
    1. Header band         — recipe name
    2. Meta row            — cook time · rating · n_ingredients · n_steps
    3. Tags                — pill-style tag row
    4. Description         — short paragraph (if present)
    5. HR divider
    6. Two-column middle:
         Left  (55%) → Ingredients list
         Right (43%) → Steps (numbered)
    7. HR divider
    8. Nutrition           — ONE of: bar / h-bar / radar / table_only
    9. HR divider
   10. Reviews             — up to MAX_REVIEWS quoted blocks
   11. Footer

Every section is conditional — if parsed value is None/empty, section is skipped.
Fit-to-page: font sizes and section spacing scale down when content is dense.
"""

import io
import math
import logging
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, Image,
    KeepTogether,
)

from config import (
    ColorScheme, VariationConfig,
    PAGE_W, PAGE_H, MARGIN_H, MARGIN_V_TOP, MARGIN_V_BOT, USABLE_W,
    CHART_KEYS, CHART_LABELS, NUTRITION_KEYS, NUTRITION_LABELS, NUTRITION_UNITS,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  FONT SCALE
#  Recipes with many steps + long reviews need slightly smaller text
#  so everything fits on one page.
# ══════════════════════════════════════════════════════════════════

def _font_scale(n_steps: int, n_reviews: int, has_description: bool) -> float:
    """
    Return a scale factor (0.78 – 1.0) based on content density.
    Applied uniformly to all body font sizes.
    """
    score = 0
    score += max(0, n_steps - 5) * 0.04
    score += max(0, n_reviews - 3) * 0.06
    score += 0.05 if has_description else 0
    return max(0.78, 1.0 - score)


# ══════════════════════════════════════════════════════════════════
#  STYLE FACTORY
# ══════════════════════════════════════════════════════════════════

def _make_styles(sc: ColorScheme, scale: float = 1.0) -> dict:
    base = getSampleStyleSheet()
    s = lambda x: max(6.0, x * scale)   # scaled, floor at 6pt

    return {
        "title": ParagraphStyle(
            "RT2", parent=base["Title"],
            fontSize=21, textColor=sc.header_text,
            leading=27, alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "section": ParagraphStyle(
            "RS2", parent=base["Heading2"],
            fontSize=s(12.5), textColor=sc.accent,
            leading=s(16), spaceBefore=s(5), spaceAfter=s(2),
            fontName="Helvetica-Bold",
        ),
        "meta": ParagraphStyle(
            "RM2", parent=base["Normal"],
            fontSize=s(10), textColor=sc.secondary_text,
            leading=s(13), alignment=TA_CENTER,
            fontName="Helvetica",
        ),
        "tag": ParagraphStyle(
            "RTAG2", parent=base["Normal"],
            fontSize=s(10), textColor=sc.accent2,
            leading=s(12), fontName="Helvetica-Bold",
        ),
        "description": ParagraphStyle(
            "RD2", parent=base["Normal"],
            fontSize=s(10), textColor=sc.body_text,
            leading=s(14), alignment=TA_JUSTIFY,
            fontName="Helvetica-Oblique", spaceAfter=s(3),
        ),
        "ingredient": ParagraphStyle(
            "RI2", parent=base["Normal"],
            fontSize=s(10), leading=s(13),
            textColor=sc.body_text,
            fontName="Helvetica", leftIndent=3, spaceAfter=1,
        ),
        "step": ParagraphStyle(
            "RSTEP2", parent=base["Normal"],
            fontSize=s(10), leading=s(13),
            textColor=sc.body_text,
            fontName="Helvetica", leftIndent=3, spaceAfter=2,
        ),
        "review_quote": ParagraphStyle(
            "RRQ2", parent=base["Normal"],
            fontSize=s(10), leading=s(13),
            textColor=sc.body_text,
            fontName="Helvetica-Oblique",
            leftIndent=6, rightIndent=6,
        ),
        "badge": ParagraphStyle(
            "RB2", fontSize=s(10),
            textColor=sc.badge_text,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "footer": ParagraphStyle(
            "RF2", fontSize=7.5,
            textColor=sc.secondary_text,
            alignment=TA_CENTER,
            fontName="Helvetica-Oblique",
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  HR helper
# ══════════════════════════════════════════════════════════════════

def _hr(sc: ColorScheme, before: float = 3, after: float = 3) -> HRFlowable:
    return HRFlowable(
        width=USABLE_W, thickness=0.6, color=sc.accent,
        spaceBefore=before, spaceAfter=after,
    )


# ══════════════════════════════════════════════════════════════════
#  SECTION BUILDERS
# ══════════════════════════════════════════════════════════════════

# ── 1. Header ─────────────────────────────────────────────────────

def _header(name: str, sc: ColorScheme, S: dict) -> list:
    tbl = Table([[Paragraph(name, S["title"])]], colWidths=[USABLE_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), sc.header_bg),
        ("BOX",           (0,0),(-1,-1), 1.0, sc.accent),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
    ]))
    return [tbl, Spacer(1, 3)]


# ── 2. Meta row ───────────────────────────────────────────────────

def _meta_row(cook_time: Optional[str], rating: Optional[str],
              n_ing: Any, n_steps: Any,
              sc: ColorScheme, S: dict) -> list:
    parts = []
    if cook_time:
        parts.append(f"⏱ {cook_time}")
    if rating:
        parts.append(rating)
    if n_ing and not _missing(n_ing):
        parts.append(f"{int(n_ing)} ingredients")
    if n_steps and not _missing(n_steps):
        parts.append(f"{int(n_steps)} steps")

    if not parts:
        return []

    text = "   •   ".join(parts)
    return [Paragraph(text, S["meta"]), Spacer(1, 3)]


def _missing(val):
    if val is None: return True
    if isinstance(val, float) and np.isnan(val): return True
    return False


# ── 3. Tags ───────────────────────────────────────────────────────

def _tags_row(tags: Optional[List[str]], sc: ColorScheme, S: dict) -> list:
    if not tags:
        return []

    # Build inline tag pills as a single paragraph with spacing
    tag_text = "  ".join(
        f'<font color="{sc.accent2.hexval() if hasattr(sc.accent2,"hexval") else "#888888"}">'
        f'[{t}]</font>'
        for t in tags
    )
    # Fallback: plain text tags separated by bullets
    plain = "  ·  ".join(tags)
    return [Paragraph(plain, S["tag"]), Spacer(1, 3)]


# ── 4. Description ────────────────────────────────────────────────

def _description(desc: Optional[str], sc: ColorScheme, S: dict) -> list:
    if not desc:
        return []
    return [Paragraph(f'"{desc}"', S["description"])]


# ── 5. Ingredients + Steps (two-column) ───────────────────────────

def _ing_steps_section(
    ingredients: Optional[List[str]],
    steps: Optional[List[str]],
    sc: ColorScheme, S: dict,
) -> list:
    story = []
    story.append(_hr(sc))

    has_ing   = bool(ingredients)
    has_steps = bool(steps)

    if not has_ing and not has_steps:
        return []

    if has_ing and has_steps:
        # Two-column
        ing_items  = [Paragraph(f"• {i}", S["ingredient"]) for i in ingredients]
        step_items = [Paragraph(s, S["step"]) for s in steps]

        left_col  = [Paragraph("Ingredients", S["section"])] + ing_items
        right_col = [Paragraph("Steps", S["section"])] + step_items

        tbl = Table(
            [[left_col, right_col]],
            colWidths=[USABLE_W * 0.44, USABLE_W * 0.54],
        )
        tbl.setStyle(TableStyle([
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ]))
        story.append(tbl)

    elif has_ing:
        story.append(Paragraph("Ingredients", S["section"]))
        for i in ingredients:
            story.append(Paragraph(f"• {i}", S["ingredient"]))

    else:
        story.append(Paragraph("Steps", S["section"]))
        for s in steps:
            story.append(Paragraph(s, S["step"]))

    return story


# ══════════════════════════════════════════════════════════════════
#  NUTRITION CHARTS  (one only — chart OR table)
# ══════════════════════════════════════════════════════════════════

def _fig_to_image(fig, w_pt: float, h_pt: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w_pt, height=h_pt)


def _chart_bar(nutri: Dict, sc: ColorScheme, w_pt: float, h_pt: float) -> Image:
    vals  = [nutri.get(k, 0) for k in CHART_KEYS]
    c_hex = sc.chart_colors

    fig, ax = plt.subplots(figsize=(w_pt/72, h_pt/72))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    bars = ax.bar(CHART_LABELS, vals, color=c_hex[:4],
                  edgecolor="white", linewidth=0.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#DDDDDD")
    ax.tick_params(labelsize=9, colors="#555")
    ax.set_title("Nutrition", fontsize=9.5, fontweight="bold",
                 color=c_hex[0], pad=4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.02,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(pad=0.3)
    return _fig_to_image(fig, w_pt, h_pt)


def _chart_hbar(nutri: Dict, sc: ColorScheme, w_pt: float, h_pt: float) -> Image:
    vals  = [nutri.get(k, 0) for k in CHART_KEYS]
    c_hex = sc.chart_colors
    y     = np.arange(len(CHART_LABELS))

    fig, ax = plt.subplots(figsize=(w_pt/72, h_pt/72))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    bars = ax.barh(y, vals, color=c_hex[:4], edgecolor="white",
                   linewidth=0.5, height=0.5)
    ax.set_yticks(y); ax.set_yticklabels(CHART_LABELS, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#DDDDDD")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("Nutrition", fontsize=9.5, fontweight="bold",
                 color=c_hex[0], pad=4)
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals)*0.02,
                bar.get_y() + bar.get_height()/2,
                f"{v:.1f}", va="center", fontsize=8)
    fig.tight_layout(pad=0.3)
    return _fig_to_image(fig, w_pt, h_pt)


# def _chart_radar(nutri: Dict, sc: ColorScheme, w_pt: float, h_pt: float) -> Image:
#     vals   = [nutri.get(k, 0) for k in CHART_KEYS]
#     n      = len(CHART_KEYS)
#     angles = np.linspace(0, 2*math.pi, n, endpoint=False).tolist()
#     vals_c   = vals   + [vals[0]]
#     angles_c = angles + [angles[0]]

#     fig, ax = plt.subplots(figsize=(w_pt/72, h_pt/72),
#                            subplot_kw={"polar": True})
#     fig.patch.set_facecolor("#FAFAFA")
#     ax.set_facecolor("#FAFAFA")
#     ax.plot(angles_c, vals_c, color=sc.chart_colors[0], linewidth=1.4)
#     ax.fill(angles_c, vals_c, color=sc.chart_colors[0], alpha=0.20)
#     ax.set_thetagrids(np.degrees(angles), CHART_LABELS, fontsize=9)
#     ax.tick_params(axis="y", labelsize=7.5)
#     ax.set_title("Nutrition", fontsize=9.5, fontweight="bold",
#                  color=sc.chart_colors[0], pad=10)
#     fig.tight_layout(pad=0.3)
#     return _fig_to_image(fig, w_pt, h_pt)


def _chart_pie(nutri: Dict, sc: ColorScheme, w_pt: float, h_pt: float) -> Image:
    """
    Generate a pie chart of the four main nutrients.
    """
    vals = [nutri.get(k, 0) for k in CHART_KEYS]
    c_hex = sc.chart_colors

    # Only include non-zero nutrients to avoid empty slices
    labels = CHART_LABELS
    non_zero = [(v, l, c) for v, l, c in zip(vals, labels, c_hex) if v > 0]
    if not non_zero:
        # Fallback if all values are zero
        non_zero = [(1.0, "No data", "#CCCCCC")]
        vals, labels, colors = zip(*non_zero)
    else:
        vals, labels, colors = zip(*non_zero)

    fig, ax = plt.subplots(figsize=(w_pt/72, h_pt/72))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    wedges, texts, autotexts = ax.pie(
        vals,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 7, 'color': '#333333'}
    )
    # Improve label readability
    for autotext in autotexts:
        autotext.set_fontsize(7)
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title("Nutrition", fontsize=8, fontweight="bold",
                 color=c_hex[0] if hasattr(c_hex, '__getitem__') else '#333333', pad=6)
    fig.tight_layout(pad=0.3)
    return _fig_to_image(fig, w_pt, h_pt)



def _nutrition_table(nutri: Dict, sc: ColorScheme) -> Table:
    hdr_s = ParagraphStyle("NTH2", fontSize=9.5, fontName="Helvetica-Bold",
                            textColor=colors.white, alignment=TA_CENTER)
    lbl_s = ParagraphStyle("NTL2", fontSize=9, fontName="Helvetica",
                            textColor=sc.body_text)
    val_s = ParagraphStyle("NTV2", fontSize=9, fontName="Helvetica-Bold",
                            textColor=sc.body_text, alignment=TA_CENTER)

    lw = USABLE_W * 0.38
    vw = USABLE_W * 0.22

    rows = [[Paragraph("Nutrient", hdr_s), Paragraph("Amount", hdr_s)]]
    for key, label, unit in zip(NUTRITION_KEYS, NUTRITION_LABELS, NUTRITION_UNITS):
        val = nutri.get(key, 0)
        display = f"{val:.1f} {unit}"
        rows.append([Paragraph(label, lbl_s), Paragraph(display, val_s)])

    tbl = Table(rows, colWidths=[lw, vw])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1, 0), sc.accent2),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [sc.row_alt, colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",     (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 3),
        ("LEFTPADDING",    (0,0),(-1,-1), 5),
        ("RIGHTPADDING",   (0,0),(-1,-1), 5),
        ("VALIGN",         (0,0),(-1,-1), "MIDDLE"),
    ]))
    # Centre on page
    wrapper = Table([[tbl]], colWidths=[USABLE_W])
    wrapper.setStyle(TableStyle([
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
    ]))
    return wrapper


def _nutrition_section(
    nutri: Dict, chart_type: str, sc: ColorScheme, S: dict,
) -> list:
    story = [_hr(sc), Paragraph("Nutrition", S["section"])]

    chart_w = USABLE_W * 0.52
    chart_h = chart_w * 0.58

    try:
        if chart_type == "bar":
            el = _chart_bar(nutri, sc, chart_w, chart_h)
        elif chart_type == "horizontal_bar":
            el = _chart_hbar(nutri, sc, chart_w, chart_h)
        elif chart_type == "radar":
            # Now uses pie chart instead of radar
            el = _chart_pie(nutri, sc, chart_w, chart_h)
        else:  # table_only
            el = _nutrition_table(nutri, sc)
    except Exception as e:
        logger.error("Nutrition render failed (%s): %s", chart_type, e)
        el = _nutrition_table(nutri, sc)   # safe fallback
        
        
    # For chart types: centre the chart image
    if chart_type != "table_only" and isinstance(el, Image):
        wrapper = Table([[el]], colWidths=[USABLE_W])
        wrapper.setStyle(TableStyle([
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ]))
        story.append(Spacer(1, 3))
        story.append(wrapper)
    else:
        story.append(Spacer(1, 3))
        story.append(el)

    return story


# ── Reviews ───────────────────────────────────────────────────────

def _reviews_section(
    reviews: Optional[List[str]], sc: ColorScheme, S: dict,
) -> list:
    if not reviews:
        return []

    story = [_hr(sc), Paragraph("Reviews", S["section"])]

    for rev in reviews:
        # Tinted quote block
        cell = Table(
            [[Paragraph(f'"{rev}"', S["review_quote"])]],
            colWidths=[USABLE_W - 8],
        )
        cell.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), sc.review_bg),
            ("BOX",           (0,0),(-1,-1), 0.4, sc.accent),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
        ]))
        story.append(cell)
        story.append(Spacer(1, 3))

    return story


# ── Footer ────────────────────────────────────────────────────────

def _footer(cfg: VariationConfig, recipe_id: Any, S: dict) -> list:
    rid = f"  •  ID: {int(recipe_id)}" if recipe_id and not _missing(recipe_id) else ""
    text = (f"Theme: {cfg.scheme.name}  •  Chart: {cfg.chart}{rid}  •  "
            "Values are approximate")
    return [
        Spacer(1, 3),
        HRFlowable(width=USABLE_W, thickness=0.4,
                   color=colors.HexColor("#BBBBBB")),
        Paragraph(text, S["footer"]),
    ]


# ══════════════════════════════════════════════════════════════════
#  PUBLIC BUILD FUNCTION
# ══════════════════════════════════════════════════════════════════

def build_recipe_card(
    name: Optional[str],
    cook_time: Optional[str],
    rating: Optional[str],
    tags: Optional[List[str]],
    description: Optional[str],
    ingredients: Optional[List[str]],
    steps: Optional[List[str]],
    nutri: Optional[Dict[str, float]],
    reviews: Optional[List[str]],
    n_ing: Any,
    n_steps: Any,
    recipe_id: Any,
    out_path: str,
    cfg: VariationConfig,
    max_reviews: Optional[int] = None,     
    scale_override: Optional[float] = None, 
) -> bool:
    """
    Build a single-page recipe card PDF.
    Returns True on success, False on failure.
    """
    # Apply review limit if provided
    if max_reviews is not None and reviews:
        reviews = reviews[:max_reviews]

    # Use override scale if provided; otherwise compute
    if scale_override is not None:
        scale = scale_override
    else:
        scale = _font_scale(
            n_steps=len(steps) if steps else 0,
            n_reviews=len(reviews) if reviews else 0,
            has_description=bool(description),
        )
        
    sc = cfg.scheme
    S = _make_styles(sc, scale)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=MARGIN_H, rightMargin=MARGIN_H,
        topMargin=MARGIN_V_TOP, bottomMargin=MARGIN_V_BOT,
    )

    story = []
    title = name or "Recipe Card"

    story.extend(_header(title, sc, S))
    story.extend(_meta_row(cook_time, rating, n_ing, n_steps, sc, S))
    story.extend(_tags_row(tags, sc, S))
    story.extend(_description(description, sc, S))
    story.extend(_ing_steps_section(ingredients, steps, sc, S))

    if nutri:
        story.extend(_nutrition_section(nutri, cfg.chart, sc, S))

    story.extend(_reviews_section(reviews, sc, S))
    story.extend(_footer(cfg, recipe_id, S))

    try:
        doc.build(story)
        return True
    except Exception as exc:
        logger.error("PDF build failed for '%s': %s", title, exc, exc_info=True)
        return False
# Multimodal Dataset Creation

## Overview

This document describes the end-to-end pipeline for how we created our multimodal recipe dataset. Each sample in the final dataset consists of two modalities:

- **A recipe card image** — a rendered visual card of the recipe (PNG).
- **A structured Markdown text** — a machine-readable recipe card covering ingredients, steps, nutrition, and reviews.

The dataset is designed for training or evaluating multimodal models (e.g., vision-language models, multimodal RAG systems) on structured culinary data.
For our case we are going to use `nvidia/llama-nemotron-embed-vl-1b-v2` embedding model.

---

## Data Source

| Property | Details |
|:---|:---|
| **Dataset** | [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/data) |
| **Raw files** | `RAW_recipes.csv`, `RAW_interactions.csv` |
| **Total raw recipes** | ~231,637 |

### Raw Schema — `RAW_recipes.csv`

Key columns used from the recipes file:

| Column | Description |
|:---|:---|
| `id` | Unique recipe identifier |
| `name` | Recipe name |
| `minutes` | Preparation time in minutes |
| `contributor_id` | Author ID (dropped) |
| `submitted` | Submission date (dropped) |
| `tags` | List of descriptive tags |
| `nutrition` | Nutritional values (calories, fat, sugar, sodium, protein, saturated fat, carbs) |
| `n_steps` | Number of preparation steps |
| `steps` | Ordered list of preparation steps |
| `description` | Recipe description text |
| `ingredients` | List of ingredient strings |
| `n_ingredients` | Number of ingredients |

### Raw Schema — `RAW_interactions.csv`

| Column | Description |
|:---|:---|
| `recipe_id` | Foreign key to recipe |
| `user_id` | Reviewer identifier |
| `rating` | Star rating (1–5) |
| `review` | Free-text review |

---

## Pipeline

The dataset creation follows five sequential stages.

```
RAW_recipes.csv ───────┐
                       ├──► [1] Load & EDA ──► [2] Merge Interactions ──► [3] Filter & Sample
RAW_interactions.csv ──┘                                                       │
                                                                               ▼
                                                         [4] Generate Markdown + Recipe Card Image
                                                                               │
                                                                               ▼
                                                              [5] Assemble Final DataFrame & Save
```

---

### Stage 1 — Load & Exploratory Analysis

Both CSVs are loaded with `pandas.read_csv`. Initial checks are performed:

- `df.isna().sum()` — identify missing values in both tables.
- `df.duplicated().sum()` — check for duplicate recipe rows.
- Distribution plots for `n_steps` and `n_ingredients` to guide filtering thresholds.

Coverage analysis is run across candidate thresholds (5, 8, 10, 12, 15, 18, 20, 25) to understand what percentage of recipes each cutoff retains:

```python
sample_n_steps = [5, 8, 10, 12, 15, 18, 20, 25]
for n in sample_n_steps:
    subset = recipes_df[recipes_df['n_steps'] <= n]
    print(f"n_steps <= {n}: {(len(subset) / len(recipes_df)) * 100:.2f}%")
```

Rows with any `NaN` are dropped from both DataFrames before merging.

---

### Stage 2 — Merge Interaction Signals

The interactions table is aggregated per recipe to compute a summary rating signal:

```python
aggregated_interactions = interactions_df.groupby('recipe_id').agg({
    'rating': 'mean',
    'review': lambda x: x.dropna().astype(str).tolist(),
    'user_id': 'count'   # becomes 'num of ratings'
}).reset_index()
```

The aggregated table is left-joined onto the recipes DataFrame on `id` → `recipe_id`. The original `id` column is dropped after the merge, and `recipe_id` is used as the canonical identifier going forward.

**Columns after merge (key additions):**

| Column | Description |
|:---|:---|
| `rating` | Mean star rating across all reviewers |
| `review` | Python list of raw review strings |
| `num of ratings` | Count of distinct user ratings |

---

### Stage 3 — Filtering & Sampling

The full merged DataFrame (~231k rows) is reduced to a focused subset using the following criteria, applied in order:

| Step | Criterion | Rationale |
|:---|:---|:---|
| 1 | Sort by `rating` descending | Prioritise higher-quality recipes |
| 2 | `n_ingredients <= 15` | Keep recipes feasible to render cleanly on a card |
| 3 | `n_steps <= 15` | Same rationale |
| 4 | `num of ratings > 10` | Require a minimum review signal for quality assurance |
| 5 | Drop remaining `NaN` rows | Final cleanliness pass |

```python
recipes_df_15k_samples = recipes_df[
    (recipes_df['n_ingredients'] <= 15) &
    (recipes_df['n_steps'] <= 15) &
    (recipes_df['num of ratings'] > 10)
].reset_index(drop=True)
```

The filtered subset is saved to `data/all csv files/recipes_15k_samples.csv` as an intermediate checkpoint.

Two columns not needed downstream are dropped at this stage:

```python
recipes_df_15k_samples.drop(columns=['contributor_id', 'submitted'], inplace=True)
```

---

### Stage 4 — Review Cleaning

Raw review strings pulled from the interactions table contain HTML entities, encoding artifacts, and very short noise strings. A `clean_reviews` function normalises each review list:

```python
def clean_reviews(review_input):
    # Parse stringified list if needed
    if isinstance(review_input, str) and review_input.startswith('['):
        review_list = ast.literal_eval(review_input)
    elif isinstance(review_input, list):
        review_list = review_input
    else:
        return []

    cleaned = []
    for review in review_list:
        review = html.unescape(review)                  # decode HTML entities
        review = review.encode('ascii', 'ignore').decode('ascii')  # strip non-ASCII
        review = re.sub(r'\s+', ' ', review).strip()    # collapse whitespace
        if len(review) > 5:                             # drop trivially short reviews
            cleaned.append(review)
    return cleaned
```

The function is applied column-wise: `recipes_df_15k_samples['review'] = recipes_df_15k_samples['review'].apply(clean_reviews)`.

---

### Stage 5 — Markdown Recipe Card Generation

Each row is converted to a structured Markdown string that serves as the **text modality** of the dataset. The `recipe_to_markdown_string` function formats all fields into a consistent schema:

```python
def recipe_to_markdown_string(row):
    ...
    markdown_text = f"""# {name}
**Recipe ID:** {r_id}
**Cook Time:** {minutes} minutes
**Rating:** {rating}/5 stars ({n_ratings} reviews)

## Description
{description}

## Ingredients
{ingredients_str}

## Instructions
{steps_str}

## Nutrition (PDV)
{nutri_str}

## Reviews
{reviews_str}
"""
    return markdown_text
```

**Nutrition labels used:** Calories, Total Fat, Sugar, Sodium, Protein, Saturated Fat (parsed from the 7-element `nutrition` list; the 7th element — Total Carbs — is omitted in the compact string).

The column is added to the DataFrame:
```python
recipes_df_15k_samples['markdown_recipe'] = recipes_df_15k_samples.apply(recipe_to_markdown_string, axis=1)
```

---

### Stage 6 — Image Modality Integration

Recipe card images are pre-generated (in a separate rendering step) and stored at:

```
data/output/images/recipe_id_<ID>.png
```

**Filtering failed renders:** 12 recipe IDs failed during image generation and are excluded:

```python
failed_recipe_ids_for_image = [3345, 4802, 5283, 6951, 6294, 8565, 11168, 11310, 13109, 14774, 15407, ...]
recipes_df_15k_samples = recipes_df_15k_samples[
    ~recipes_df_15k_samples['recipe_id'].isin(failed_recipe_ids_for_image)
]
```

**Image path column:** A new column `image_path` is added by checking whether the corresponding PNG exists on disk:

```python
image_dir = "data/output/images"
recipes_df_15k_samples['image_path'] = recipes_df_15k_samples['recipe_id'].apply(
    lambda x: f"{image_dir}/recipe_id_{x}.png"
    if os.path.exists(os.path.join(image_dir, f"recipe_id_{x}.png"))
    else "no image"
)

# Drop any rows where the image is genuinely missing
recipes_df_15k_samples = recipes_df_15k_samples[recipes_df_15k_samples['image_path'] != "no image"]
```

`recipe_id` is cast to `int` before path construction to avoid float-based filenames (e.g., `recipe_id_123.0.png`).

---

## Final Dataset Schema

The final CSV is saved to `data/all csv files/recipes_15k_samples.csv`.

| Column | Type | Description |
|:---|:---|:---|
| `name` | `str` | Recipe name |
| `recipe_id` | `int` | Unique recipe identifier |
| `minutes` | `int` | Total cook/prep time in minutes |
| `tags` | `list[str]` | Descriptive tags (stored as stringified list) |
| `nutrition` | `list[float]` | Nutritional values — [Calories, Fat, Sugar, Sodium, Protein, Sat. Fat, Carbs] |
| `n_steps` | `int` | Number of preparation steps (≤ 15) |
| `steps` | `list[str]` | Ordered preparation steps |
| `description` | `str` | Recipe description |
| `ingredients` | `list[str]` | List of ingredients |
| `n_ingredients` | `int` | Number of ingredients (≤ 15) |
| `recipe_id` (merged) | `int` | Foreign key from interactions |
| `rating` | `float` | Mean star rating (> 0) |
| `review` | `list[str]` | Cleaned review texts (top 5 used in markdown) |
| `num of ratings` | `int` | Total number of reviewer interactions (> 10) |
| `markdown_recipe` | `str` | Full structured Markdown recipe card (text modality) |
| `image_path` | `str` | Relative path to the rendered PNG recipe card (image modality) |

---

## Data Quality Summary

| Metric | Value |
|:---|:---|
| Raw recipes | ~231,637 |
| After filtering (n_steps ≤ 15, n_ingredients ≤ 15, ratings > 10) | ~15k range |
| Failed image renders excluded | 12 |
| Final dataset size | ~15k samples (exact count depends on image availability) |
| Modalities per sample | 2 (PNG image + Markdown text) |

---

## File Structure

```
data/
├── all csv files/
│   ├── RAW_recipes.csv             # Original Kaggle recipes
│   ├── RAW_interactions.csv        # Original Kaggle interactions
│   └── recipes_15k_samples.csv     # Final multimodal dataset CSV
└── output/
    └── images/
        └── recipe_id_<ID>.png      # Pre-rendered recipe card images
```

---

## Dependencies

| Library | Purpose |
|:---|:---|
| `pandas` | Data loading, merging, filtering, serialisation |
| `ast` | Safe parsing of stringified Python lists from CSV |
| `re`, `html` | Review text cleaning |
| `os` | File existence checks for image path validation |
| `transformers` (`load_image`) | Sanity-check image loading during development |
| `IPython.display` | Notebook-side Markdown rendering during EDA |
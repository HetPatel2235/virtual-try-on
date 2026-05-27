"""Garment category helpers for cloud try-on (IDM-VTON) and validation."""

from __future__ import annotations

# IDM-VTON garment_des values
CLOUD_UPPER = "Upper body"
CLOUD_LOWER = "Lower body"
CLOUD_DRESS = "Dresses"

_LOWER_EXACT = frozenset({
    "lower body", "lower_body", "lower",
    "pants", "pant", "jeans", "jean",
    "cargo", "cargos", "cargo pants", "cargo_pants",
    "trousers", "trouser", "chinos", "chino",
    "shorts", "short", "skirt", "skirts",
    "bottoms", "bottom", "joggers", "leggings",
})

_UPPER_EXACT = frozenset({
    "upper body", "upper_body", "upper",
    "tshirt", "t-shirt", "t_shirt", "shirt", "jacket", "tops", "top",
    "hoodie", "sweater", "blouse",
})

_DRESS_EXACT = frozenset({"dress", "dresses"})

_LOWER_KEYWORDS = ("pant", "jean", "cargo", "short", "skirt", "chino", "trouser", "bottom", "jogger", "legging")


def is_lower_body_category(category: str) -> bool:
    c = category.lower().strip()
    if c in _LOWER_EXACT:
        return True
    return any(kw in c for kw in _LOWER_KEYWORDS)


def is_upper_body_category(category: str) -> bool:
    c = category.lower().strip()
    if c in _UPPER_EXACT:
        return True
    return any(kw in c for kw in ("shirt", "jacket", "top", "hoodie", "sweater", "blouse"))


def is_dress_category(category: str) -> bool:
    return category.lower().strip() in _DRESS_EXACT


def is_valid_tryon_category(category: str) -> bool:
    """Categories accepted by the cloud try-on pipeline."""
    if not category or not category.strip():
        return False
    return (
        is_lower_body_category(category)
        or is_upper_body_category(category)
        or is_dress_category(category)
    )


def category_for_cloud(category: str) -> str:
    """Map any catalog/UI category string to IDM-VTON garment_des."""
    if is_dress_category(category):
        return CLOUD_DRESS
    if is_lower_body_category(category):
        return CLOUD_LOWER
    return CLOUD_UPPER


def catalog_is_lower(category: str) -> bool:
    """Whether a garment metadata category belongs in the Lowers catalog tab."""
    return is_lower_body_category(category)


def garment_prompt_for_cloud(category: str, garment_name: str = "") -> str:
    """
    IDM-VTON garment_des is a text prompt (not a body-region selector).
    Use a descriptive garment phrase for the diffusion model.
    """
    if garment_name and garment_name.strip():
        return garment_name.strip()

    c = category.lower().strip()
    if is_lower_body_category(category):
        if "jean" in c:
            return "blue slim fit denim jeans"
        if "cargo" in c:
            return "olive green cargo pants with side pockets"
        if "chino" in c:
            return "navy tapered chino trousers"
        if "short" in c:
            return "casual shorts"
        if "jogger" in c:
            return "grey jogger sweatpants"
        if "skirt" in c:
            return "midi skirt"
        return "pants trousers"
    if is_dress_category(category):
        return "dress"
    if "jacket" in c:
        return "casual jacket"
    if "hoodie" in c:
        return "hoodie sweatshirt"
    if "shirt" in c and "t-shirt" not in c and "tshirt" not in c:
        return "button-down shirt"
    return "short sleeve t-shirt"

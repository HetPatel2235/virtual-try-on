"""Tests for garment category normalization."""

from ml_ai.core.garment_categories import (
    catalog_is_lower,
    category_for_cloud,
    garment_prompt_for_cloud,
    is_valid_tryon_category,
    CLOUD_LOWER,
    CLOUD_UPPER,
    CLOUD_DRESS,
)


def test_lower_body_categories():
    for cat in ("jeans", "cargo pants", "pants", "Lower body", "shorts"):
        assert catalog_is_lower(cat)
        assert is_valid_tryon_category(cat)
        assert category_for_cloud(cat) == CLOUD_LOWER


def test_upper_body_categories():
    for cat in ("tshirt", "shirt", "jacket", "Upper body"):
        assert not catalog_is_lower(cat)
        assert is_valid_tryon_category(cat)
        assert category_for_cloud(cat) == CLOUD_UPPER


def test_dress_category():
    assert category_for_cloud("dress") == CLOUD_DRESS
    assert is_valid_tryon_category("dress")


def test_invalid_category():
    assert not is_valid_tryon_category("")
    assert not is_valid_tryon_category("socks")


def test_garment_prompt_not_body_region():
    assert "body" not in garment_prompt_for_cloud("jeans").lower()
    assert "denim" in garment_prompt_for_cloud("jeans").lower()
    assert garment_prompt_for_cloud("jeans", "My Custom Jeans") == "My Custom Jeans"

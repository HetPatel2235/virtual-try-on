"""
Cloud Engine module using Hugging Face Gradio API.
Offloads heavy Deep Learning inference to IDM-VTON.
"""
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
from gradio_client import Client, handle_file

logger = logging.getLogger(__name__)

# Minimum mean pixel change in the relevant body region (0–255 scale)
MIN_CHANGE_UPPER = 4.0
MIN_CHANGE_LOWER = 6.0


class CloudTryOnEngine:
    def __init__(self, api_url: str = "yisol/IDM-VTON"):
        self.api_url = api_url.strip() if api_url else "yisol/IDM-VTON"
        self.client = Client(self.api_url)
        logger.info(f"Connected to Cloud API: {self.api_url}")

    def run(
        self,
        person_path: str,
        garment_path: str,
        garment_category: str = "tshirt",
        garment_name: str = "",
    ) -> dict:
        """
        Sends images to Hugging Face space or custom Colab API.

        Public HF space only auto-masks the upper body (is_checked=True).
        Lower-body garments use a painted leg mask (is_checked=False).
        """
        import time

        from ml_ai.core.garment_categories import (
            garment_prompt_for_cloud,
            is_lower_body_category,
        )
        from ml_ai.core.tryon_masks import build_image_editor_layer

        t_start = time.perf_counter()
        warnings: list[str] = []

        prompt = garment_prompt_for_cloud(garment_category, garment_name)
        is_lower = is_lower_body_category(garment_category)
        is_checked = not is_lower
        layer_files = []
        temp_layer_path = None

        person_bgr = cv2.imread(person_path)
        if person_bgr is None:
            return {
                "composite_image": None,
                "processing_time_s": 0.0,
                "success": False,
                "error": f"Cannot read person image: {person_path}",
                "warnings": [],
            }

        if is_lower:
            logger.info("Lower body: building leg mask layer for ImageEditor.")
            layer_bgr, mask_gray, legs_visible = build_image_editor_layer(person_bgr)
            if not legs_visible:
                warnings.append(
                    "Legs may not be visible in your photo. Use a full-body front-facing "
                    "photo (hips to feet) for pants try-on."
                )
            coverage = (mask_gray > 127).mean() * 100
            if coverage < 5.0:
                return {
                    "composite_image": None,
                    "processing_time_s": 0.0,
                    "success": False,
                    "error": "Could not build a leg mask. Upload a full-body photo with legs visible.",
                    "warnings": warnings,
                }

            layer_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_layer_path = layer_file.name
            layer_file.close()
            cv2.imwrite(temp_layer_path, layer_bgr)
            layer_files = [handle_file(temp_layer_path)]

            debug_dir = Path("database/data/tryon_debug")
            if debug_dir.parent.exists():
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_dir / "debug_lower_mask_layer.png"), layer_bgr)
                cv2.imwrite(str(debug_dir / "debug_lower_mask_gray.png"), mask_gray)

        dict_img = {
            "background": handle_file(person_path),
            "layers": layer_files,
            "composite": None,
        }

        try:
            logger.info(
                f"IDM-VTON prompt={prompt!r}, auto_mask={is_checked}, lower={is_lower}"
            )
            result = self.client.predict(
                dict=dict_img,
                garm_img=handle_file(garment_path),
                garment_des=prompt,
                is_checked=is_checked,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon",
            )

            out_path = result[0]
            composite = cv2.imread(out_path)
            if composite is None:
                raise ValueError("Failed to read output image from API.")

            elapsed = time.perf_counter() - t_start

            change = _region_change_score(person_bgr, composite, lower_body=is_lower)
            logger.info(f"Try-on region change score: {change:.2f}")

            min_change = MIN_CHANGE_LOWER if is_lower else MIN_CHANGE_UPPER
            if change < min_change:
                msg = (
                    "The cloud try-on returned an image that looks almost identical to your "
                    "original photo. "
                )
                if is_lower:
                    msg += (
                        "For pants/jeans: use a full-body photo and a flat-lay garment image. "
                        "If this keeps happening, run IDM_VTON_Worker.ipynb on Colab and paste "
                        "your .gradio.live URL in Cloud Worker Settings."
                    )
                else:
                    msg += (
                        "Try a clearer garment photo or a personal Colab GPU link under "
                        "Cloud Worker Settings."
                    )
                return {
                    "composite_image": composite,
                    "processing_time_s": round(elapsed, 3),
                    "success": False,
                    "error": msg,
                    "warnings": warnings,
                }

            if is_lower and change < MIN_CHANGE_LOWER + 4:
                warnings.append(
                    "Lower-body change is subtle. Use a real flat-lay pants photo (not a "
                    "placeholder) and a full-body person photo for best results."
                )

            return {
                "composite_image": composite,
                "processing_time_s": round(elapsed, 3),
                "success": True,
                "error": None,
                "warnings": warnings,
            }

        except Exception as e:
            elapsed = time.perf_counter() - t_start
            logger.error(f"Cloud API Error: {e}")
            return {
                "composite_image": None,
                "processing_time_s": round(elapsed, 3),
                "success": False,
                "error": str(e),
                "warnings": warnings,
            }
        finally:
            if temp_layer_path:
                try:
                    Path(temp_layer_path).unlink(missing_ok=True)
                except Exception:
                    pass


def _region_change_score(
    person_bgr: np.ndarray,
    output_bgr: np.ndarray,
    lower_body: bool,
) -> float:
    """Mean absolute pixel difference in the body region that should change."""
    out = output_bgr
    orig = cv2.resize(person_bgr, (out.shape[1], out.shape[0]))
    h = out.shape[0]
    if lower_body:
        region = slice(int(h * 0.30), h)
    else:
        region = slice(0, int(h * 0.55))
    a = orig[region].astype(np.float32)
    b = out[region].astype(np.float32)
    return float(np.mean(np.abs(a - b)))

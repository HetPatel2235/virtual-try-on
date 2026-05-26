"""
Cloud Engine module using Hugging Face Gradio API.
Offloads heavy Deep Learning inference to IDM-VTON.
"""
from gradio_client import Client, handle_file
import cv2
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class CloudTryOnEngine:
    def __init__(self, api_url: str = "yisol/IDM-VTON"):
        self.api_url = api_url.strip() if api_url else "yisol/IDM-VTON"
        self.client = Client(self.api_url)
        logger.info(f"Connected to Cloud API: {self.api_url}")

    def run(self, person_path: str, garment_path: str, garment_category: str = "Upper body") -> dict:
        """
        Sends images to Hugging Face space or custom Colab API and returns the result.
        Returns a dict with:
        - 'composite_image': numpy array
        - 'processing_time_s': float
        - 'success': bool
        - 'error': str (if failed)
        """
        t_start = time.perf_counter()
        
        # IDM-VTON expects image editor dict for person
        dict_img = {
            "background": handle_file(person_path),
            "layers": [],
            "composite": None
        }

        try:
            # Standardize category for IDM-VTON (which expects Title Case strings)
            std_category = "Upper body"
            cat_lower = garment_category.lower().strip()
            if cat_lower in ["lower body", "lower_body", "pants", "skirt", "shorts"]:
                std_category = "Lower body"
            elif cat_lower in ["dress", "dresses"]:
                std_category = "Dresses"

            logger.info(f"Sending request to IDM-VTON (Category: {std_category})...")
            result = self.client.predict(
                dict=dict_img,
                garm_img=handle_file(garment_path),
                garment_des=std_category,
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            
            # result[0] is the path to the output image
            # result[1] is the masked image output
            out_path = result[0]
            
            # Read image back to numpy array
            composite = cv2.imread(out_path)
            if composite is None:
                raise ValueError("Failed to read output image from API.")

            elapsed = time.perf_counter() - t_start
            
            return {
                "composite_image": composite,
                "processing_time_s": round(elapsed, 3),
                "success": True,
                "error": None
            }

        except Exception as e:
            elapsed = time.perf_counter() - t_start
            logger.error(f"Cloud API Error: {e}")
            return {
                "composite_image": None,
                "processing_time_s": round(elapsed, 3),
                "success": False,
                "error": str(e)
            }

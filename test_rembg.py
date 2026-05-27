import cv2
import numpy as np
from PIL import Image
import rembg

def test_rembg():
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (75, 75), (0, 255, 0), -1)
    
    # Convert BGR to PIL
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    
    # Remove background
    no_bg = rembg.remove(pil_img)
    
    # Composite onto a clean studio gray background
    studio_bg = Image.new("RGBA", no_bg.size, (240, 240, 240, 255))
    studio_bg.paste(no_bg, (0, 0), no_bg)
    person_img = cv2.cvtColor(np.array(studio_bg), cv2.COLOR_RGBA2BGR)
    
    print(f"Success! Output shape: {person_img.shape}")

if __name__ == "__main__":
    test_rembg()

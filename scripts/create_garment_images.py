"""Create placeholder garment images"""

import cv2
import numpy as np
from pathlib import Path

garments = ["tshirt-001", "tshirt-002", "shirt-001"]

for garment_id in garments:
    path = Path(f"database/data/garments/{garment_id}")
    path.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder image (512x512, white background)
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Draw a simple garment shape (rectangle)
    cv2.rectangle(img, (100, 50), (412, 450), (200, 200, 200), -1)
    
    # Save
    cv2.imwrite(str(path / "image.png"), img)
    print(f"✅ Created {garment_id}/image.png")

print("✅ All garment images created!")

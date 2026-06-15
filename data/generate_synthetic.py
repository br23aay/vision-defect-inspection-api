"""
generate_synthetic.py
---------------------
Generates a synthetic product image dataset for defect detection.
Creates two classes: 'pass' (clean) and 'fail' (defective).
No real data required — runs entirely on CPU with NumPy + Pillow.

Usage:
    python data/generate_synthetic.py
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUTPUT_DIR = "data/images"
CLASSES = ["pass", "fail"]
SAMPLES_PER_CLASS = 150
IMG_SIZE = (224, 224)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def generate_clean_product(size=IMG_SIZE):
    """Generate a clean product image — uniform surface, no defects."""
    base_colour = (
        random.randint(180, 220),
        random.randint(180, 220),
        random.randint(180, 220),
    )
    img_array = np.ones((size[1], size[0], 3), dtype=np.uint8)
    img_array[:] = base_colour

    # Add subtle texture noise
    noise = np.random.randint(-10, 10, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    draw = ImageDraw.Draw(img)
    # Draw a clean product border
    draw.rectangle(
        [10, 10, size[0] - 10, size[1] - 10],
        outline=(100, 100, 100),
        width=2,
    )
    return img


def generate_defective_product(size=IMG_SIZE):
    """Generate a defective product image — scratches, spots, cracks."""
    img = generate_clean_product(size)
    draw = ImageDraw.Draw(img)

    defect_type = random.choice(["scratch", "spot", "crack", "multi"])

    if defect_type in ("scratch", "multi"):
        # Draw scratch lines
        for _ in range(random.randint(1, 4)):
            x1, y1 = random.randint(20, size[0] - 20), random.randint(20, size[1] - 20)
            x2, y2 = x1 + random.randint(-80, 80), y1 + random.randint(-80, 80)
            colour = (random.randint(50, 120), random.randint(50, 120), random.randint(50, 120))
            draw.line([x1, y1, x2, y2], fill=colour, width=random.randint(1, 3))

    if defect_type in ("spot", "multi"):
        # Draw dark spots / contamination
        for _ in range(random.randint(1, 5)):
            cx, cy = random.randint(20, size[0] - 20), random.randint(20, size[1] - 20)
            r = random.randint(3, 15)
            colour = (random.randint(30, 100), random.randint(30, 100), random.randint(30, 100))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)

    if defect_type in ("crack", "multi"):
        # Draw irregular crack pattern
        points = [(random.randint(20, size[0] - 20), random.randint(20, size[1] - 20))]
        for _ in range(random.randint(3, 8)):
            last = points[-1]
            points.append(
                (last[0] + random.randint(-30, 30), last[1] + random.randint(-30, 30))
            )
        draw.line(points, fill=(40, 40, 40), width=2)

    return img


def generate_dataset():
    """Generate and save the full synthetic dataset."""
    for cls in CLASSES:
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)

    print(f"Generating {SAMPLES_PER_CLASS} images per class...")

    for i in range(SAMPLES_PER_CLASS):
        # PASS — clean product
        img = generate_clean_product()
        img.save(os.path.join(OUTPUT_DIR, "pass", f"pass_{i:04d}.jpg"), "JPEG", quality=90)

        # FAIL — defective product
        img = generate_defective_product()
        img.save(os.path.join(OUTPUT_DIR, "fail", f"fail_{i:04d}.jpg"), "JPEG", quality=90)

    total = SAMPLES_PER_CLASS * 2
    print(f"Dataset generated: {total} images in {OUTPUT_DIR}/")
    print(f"  pass/: {SAMPLES_PER_CLASS} images")
    print(f"  fail/: {SAMPLES_PER_CLASS} images")


if __name__ == "__main__":
    generate_dataset()

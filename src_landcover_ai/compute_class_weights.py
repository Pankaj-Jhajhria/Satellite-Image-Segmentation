"""Run this once, before train.py, to see how skewed the classes actually
are and get weights to correct for it. Sampling 500 tiles (not all ~7500)
is plenty for a stable estimate and keeps this fast.

Run from the project root:
    python src_landcover_ai/compute_class_weights.py
"""
import os
import sys
import random
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TILES_DIR, TRAIN_LIST, NUM_CLASSES, CLASS_COLORS

SAMPLE_SIZE = 500

with open(TRAIN_LIST) as f:
    names = [line.strip() for line in f if line.strip()]

random.seed(42)
sample = random.sample(names, min(SAMPLE_SIZE, len(names)))

pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

for name in sample:
    mask_path = os.path.join(TILES_DIR, f"{name}_m.png")
    mask = np.array(Image.open(mask_path).convert("L"))
    for class_id in range(NUM_CLASSES):
        pixel_counts[class_id] += np.sum(mask == class_id)

total_pixels = pixel_counts.sum()

print(f"sampled {len(sample)} tiles\n")
print(f"{'class':<12}{'pixels':>14}{'share':>10}")
for class_id in range(NUM_CLASSES):
    class_name = CLASS_COLORS[class_id][0]
    share = pixel_counts[class_id] / total_pixels
    print(f"{class_name:<12}{pixel_counts[class_id]:>14}{share:>9.2%}")

# inverse-frequency weights, normalized so they average to 1.0 -
# a class with half the average frequency gets ~2x the weight, etc.
frequencies = pixel_counts / total_pixels
raw_weights = 1.0 / (frequencies + 1e-6)
normalized_weights = raw_weights / raw_weights.mean()

print("\npaste this into config.py as CLASS_WEIGHTS:")
print("[" + ", ".join(f"{w:.4f}" for w in normalized_weights) + "]")

"""Run this once, before train.py, to see how skewed the classes actually
are - and compare several different loss-weighting formulas side by side,
so you're picking CLASS_WEIGHTS by an informed choice, not guessing.

Sampling 500 tiles (not all ~7500) is plenty for a stable estimate and
keeps this fast.

Run from the project root:
    python src_landcover_model3/compute_class_weights.py
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
frequencies = pixel_counts / total_pixels

print(f"sampled {len(sample)} tiles\n")
print(f"{'class':<12}{'pixels':>14}{'share':>10}")
for class_id in range(NUM_CLASSES):
    class_name = CLASS_COLORS[class_id][0]
    print(f"{class_name:<12}{pixel_counts[class_id]:>14}{frequencies[class_id]:>9.2%}")


def normalize(weights):
    """Scale so the weights average to 1.0 - keeps overall loss magnitude
    comparable across formulas, only the RELATIVE gap between classes differs."""
    return weights / weights.mean()


# --- inverse frequency: what model 2 used - most aggressive ---
inv_freq = normalize(1.0 / (frequencies + 1e-6))

# --- inverse sqrt frequency: dampened version of the above ---
inv_sqrt = normalize(1.0 / np.sqrt(frequencies + 1e-6))

# --- median-frequency balancing: a classic from segmentation literature ---
median_freq = normalize(np.median(frequencies) / (frequencies + 1e-6))

# --- log-dampened: gentlest of the four, used in some real-time models ---
log_damp = normalize(1.0 / np.log(1.02 + frequencies))

print("\ncompare the formulas (all normalized to average 1.0):\n")
header = f"{'class':<12}{'inv_freq':>12}{'inv_sqrt':>12}{'median_freq':>14}{'log_damp':>12}"
print(header)
for class_id in range(NUM_CLASSES):
    class_name = CLASS_COLORS[class_id][0]
    print(f"{class_name:<12}"
          f"{inv_freq[class_id]:>12.4f}"
          f"{inv_sqrt[class_id]:>12.4f}"
          f"{median_freq[class_id]:>14.4f}"
          f"{log_damp[class_id]:>12.4f}")

print("\npaste whichever one you pick into config.py as CLASS_WEIGHTS:\n")
print("inv_freq:    [" + ", ".join(f"{w:.4f}" for w in inv_freq) + "]")
print("inv_sqrt:    [" + ", ".join(f"{w:.4f}" for w in inv_sqrt) + "]")
print("median_freq: [" + ", ".join(f"{w:.4f}" for w in median_freq) + "]")
print("log_damp:    [" + ", ".join(f"{w:.4f}" for w in log_damp) + "]")

print("\nnote: inv_freq is what model 2 used and caused the building/road\n"
      "speckling seen in visualize_predictions.py - inv_sqrt or log_damp\n"
      "are worth starting with here.")

"""Run this FIRST, before train.py. It just opens a few mask tiles and
prints the unique pixel values found in them. This whole pipeline assumes
LandCover.ai masks store the class id directly as the pixel value
(0=background, 1=building, 2=woodland, 3=water, 4=road) - values 0 to 4
and nothing else. If you see anything different (e.g. 0-255 or only 0/255),
the mask encoding is not what preprocessing.py expects, and dataset.py /
preprocessing.py will need adjusting before you train.

Run from the project root:
    python src_landcover_ai/check_mask_values.py
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TILES_DIR, TRAIN_LIST

with open(TRAIN_LIST) as f:
    names = [line.strip() for line in f if line.strip()][:5]

for name in names:
    mask_path = os.path.join(TILES_DIR, f"{name}_m.png")
    mask = np.array(Image.open(mask_path).convert("L"))
    print(f"{name}_m.png  ->  unique values: {sorted(np.unique(mask).tolist())}")

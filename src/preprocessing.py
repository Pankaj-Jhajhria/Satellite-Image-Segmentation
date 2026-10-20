import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

# make sure this file's own folder (src/) is importable, no matter how
# or from where this script was launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CLASS_DICT_CSV, IMAGE_SIZE


def load_class_colors():
    """Read class_dict.csv and return a list of (class_name, (r, g, b)),
    in the same order as the rows in the csv. The row order becomes the
    class id order (row 0 = class 0, row 1 = class 1, ...)."""
    df = pd.read_csv(CLASS_DICT_CSV)
    classes = []
    for _, row in df.iterrows():
        classes.append((row["name"], (int(row["r"]), int(row["g"]), int(row["b"]))))
    return classes


CLASS_COLORS = load_class_colors()  # loaded once, reused everywhere


def resize_image(image: Image.Image) -> Image.Image:
    """Resize a normal photo. Bilinear is fine here because colors
    are continuous - blending two colors still makes sense."""
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def resize_mask(mask: Image.Image) -> Image.Image:
    """Resize a mask. Must use NEAREST, never bilinear/bicubic - a mask
    stores class labels, and blending two labels makes no sense at all."""
    return mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)


def normalize_image(image_array: np.ndarray) -> np.ndarray:
    """Scale pixel values from the 0-255 range down to 0-1."""
    return image_array.astype(np.float32) / 255.0


def rgb_mask_to_class_ids(mask_array: np.ndarray) -> np.ndarray:
    """Turn an RGB mask (H, W, 3) into a class-id mask (H, W) of integers,
    using the colors listed in class_dict.csv."""
    height, width, _ = mask_array.shape
    class_id_mask = np.zeros((height, width), dtype=np.int64)

    for class_index, (name, color) in enumerate(CLASS_COLORS):
        matches_this_class = np.all(mask_array == np.array(color), axis=-1)
        class_id_mask[matches_this_class] = class_index

    return class_id_mask

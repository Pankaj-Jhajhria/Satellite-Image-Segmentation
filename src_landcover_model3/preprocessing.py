import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import IMAGE_SIZE, NUM_CLASSES


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


def clamp_mask_ids(mask_array: np.ndarray) -> np.ndarray:
    """Unlike DeepGlobe, LandCover.ai masks are single-channel and the
    pixel value itself IS the class id (0=background, 1=building,
    2=woodland, 3=water, 4=road) - so there's no RGB-to-id color lookup
    needed here, just a cast plus a safety clamp in case resizing or a
    stray pixel produced something outside the valid class range."""
    mask_array = mask_array.astype(np.int64)
    mask_array[(mask_array < 0) | (mask_array >= NUM_CLASSES)] = 0
    return mask_array

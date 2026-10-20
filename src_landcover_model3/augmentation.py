import random
import numpy as np


def random_flip(image: np.ndarray, mask: np.ndarray):
    """Randomly flip image and mask TOGETHER, so they stay aligned.
    image: (H, W, 3) float array
    mask:  (H, W) int array
    """
    if random.random() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

    if random.random() < 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()

    return image, mask

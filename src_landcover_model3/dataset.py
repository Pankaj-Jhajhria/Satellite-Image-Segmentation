import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# make sure this file's own folder (src_landcover_model3/) is importable, no
# matter how or from where this script was launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TILES_DIR, TRAIN_LIST, VAL_LIST, TEST_LIST
from preprocessing import resize_image, resize_mask, normalize_image, clamp_mask_ids
from augmentation import random_flip

# maps our "split" argument onto the list files split.py generated
SPLIT_FILES = {
    "train": TRAIN_LIST,
    "valid": VAL_LIST,   # split.py calls this "val", we keep "valid" to
                          # match the naming used elsewhere in this project
    "test": TEST_LIST,
}


class SegmentationDataset(Dataset):
    """Loads ONE (image, mask) pair for a given index, for the LandCover.ai
    dataset as produced by the official split.py script:
        output/<name>.jpg    - the image tile
        output/<name>_m.png  - the mask tile (single channel, pixel value
                                IS the class id - see preprocessing.py)
    split must be 'train', 'valid', or 'test'.

    Unlike the DeepGlobe pipeline, val.txt here DOES have real ground-truth
    masks (LandCover.ai ships full labels for every split), so there's no
    missing-mask / fake-zero-mask problem to work around."""

    def __init__(self, split, augment=False):
        list_path = SPLIT_FILES[split]
        with open(list_path) as f:
            self.names = [line.strip() for line in f if line.strip()]
        self.split = split
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        image_path = os.path.join(TILES_DIR, f"{name}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = resize_image(image)
        image_array = normalize_image(np.array(image))

        mask_path = os.path.join(TILES_DIR, f"{name}_m.png")
        mask = Image.open(mask_path).convert("L")  # single channel, values 0-4
        mask = resize_mask(mask)
        mask_array = clamp_mask_ids(np.array(mask))

        if self.augment:
            image_array, mask_array = random_flip(image_array, mask_array)

        # PIL/numpy gives us (H, W, 3) - pytorch wants channels first (3, H, W)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_array).long()

        return image_tensor, mask_tensor

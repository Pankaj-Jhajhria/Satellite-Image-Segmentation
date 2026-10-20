import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

# make sure this file's own folder (src/) is importable, no matter how
# or from where this script was launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, METADATA_CSV
from preprocessing import resize_image, resize_mask, normalize_image, rgb_mask_to_class_ids
from augmentation import random_flip


class SegmentationDataset(Dataset):
    """Loads ONE (image, mask) pair for a given index.
    split must be 'train', 'valid', or 'test' - matches the 'split'
    column in metadata.csv."""

    def __init__(self, split, augment=False):
        metadata = pd.read_csv(METADATA_CSV)
        self.rows = metadata[metadata["split"] == split].reset_index(drop=True)
        self.split = split
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]

        image_path = os.path.join(DATA_DIR, row["sat_image_path"])
        image = Image.open(image_path).convert("RGB")
        image = resize_image(image)
        image_array = normalize_image(np.array(image))

        # the test split has no ground-truth mask, so we fake an empty one
        has_mask = self.split != "test" and isinstance(row.get("mask_path"), str)
        if has_mask:
            mask_path = os.path.join(DATA_DIR, row["mask_path"])
            mask = Image.open(mask_path).convert("RGB")
            mask = resize_mask(mask)
            mask_array = rgb_mask_to_class_ids(np.array(mask))
        else:
            mask_array = np.zeros(image_array.shape[:2], dtype=np.int64)

        if self.augment:
            image_array, mask_array = random_flip(image_array, mask_array)

        # PIL/numpy gives us (H, W, 3) - pytorch wants channels first (3, H, W)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_array).long()

        return image_tensor, mask_tensor

import os
import sys
import matplotlib.pyplot as plt

# make sure this file's own folder (src/) is importable, no matter how
# or from where this script was launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import SegmentationDataset
from predict import class_ids_to_rgb


def show_samples(split="train", num_samples=3):
    """Plot image / ground-truth mask side by side, to sanity-check the data
    before (or after) training."""
    dataset = SegmentationDataset(split=split, augment=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))

    for i in range(num_samples):
        image_tensor, mask_tensor = dataset[i]

        image = image_tensor.permute(1, 2, 0).numpy()  # (3,H,W) -> (H,W,3)
        mask_rgb = class_ids_to_rgb(mask_tensor.numpy())

        axes[i][0].imshow(image)
        axes[i][0].set_title("image")
        axes[i][0].axis("off")

        axes[i][1].imshow(mask_rgb)
        axes[i][1].set_title("mask")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig("sample_check.png")
    print("saved sample_check.png")


if __name__ == "__main__":
    show_samples(split="train", num_samples=3)

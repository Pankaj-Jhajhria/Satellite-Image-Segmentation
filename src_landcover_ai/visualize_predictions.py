"""Picks a handful of random test.txt tiles and, for each, shows three
SEPARATE panels side by side (no blending/overlaying):
    1. the actual photo
    2. the model's predicted mask
    3. the real ground-truth label mask
This makes it easy to compare shapes and colors directly between columns
2 and 3 - did the model find a building-shaped red blob roughly where
the real label has one, get the wrong class, or miss it entirely.

Run from the project root, any time after predict.py / a training run:
    python src_landcover_ai/visualize_predictions.py
    python src_landcover_ai/visualize_predictions.py --num_samples 10
    python src_landcover_ai/visualize_predictions.py --seed 42   # reproducible picks
"""
import os
import sys
import random
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, VISUALIZE_DIR, CLASS_COLORS
from dataset import SegmentationDataset
from predict import load_model, class_ids_to_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=6,
                         help="how many random test tiles to show")
    parser.add_argument("--seed", type=int, default=None,
                         help="fix this to get the same random tiles every run")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(VISUALIZE_DIR, exist_ok=True)

    model = load_model()
    test_dataset = SegmentationDataset(split="test", augment=False)

    num_samples = min(args.num_samples, len(test_dataset))
    indices = random.sample(range(len(test_dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, 3)  # keep indexing consistent for a single row

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image_tensor, mask_tensor = test_dataset[idx]
            input_batch = image_tensor.unsqueeze(0).to(DEVICE)

            logits = model(input_batch)
            predicted_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

            image_np = image_tensor.permute(1, 2, 0).numpy()  # (H,W,3) float 0-1
            gt_mask = mask_tensor.numpy()

            predicted_rgb = class_ids_to_rgb(predicted_mask)
            gt_rgb = class_ids_to_rgb(gt_mask)

            name = test_dataset.names[idx]

            axes[row][0].imshow(image_np)
            axes[row][0].set_ylabel(name, fontsize=8)
            axes[row][0].set_xticks([])
            axes[row][0].set_yticks([])

            axes[row][1].imshow(predicted_rgb)
            axes[row][1].axis("off")

            axes[row][2].imshow(gt_rgb)
            axes[row][2].axis("off")

    axes[0][0].set_title("actual (image)", fontsize=9)
    axes[0][1].set_title("predicted", fontsize=9)
    axes[0][2].set_title("label (ground truth)", fontsize=9)

    legend_patches = [mpatches.Patch(color=np.array(color) / 255, label=name)
                       for name, color in CLASS_COLORS]
    fig.legend(handles=legend_patches, loc="upper center", ncol=len(CLASS_COLORS),
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    out_path = os.path.join(VISUALIZE_DIR, "test_predictions_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
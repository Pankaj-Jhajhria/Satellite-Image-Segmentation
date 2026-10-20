import os
import sys
import numpy as np
import torch
from PIL import Image

# make sure this file's own folder (src/) is importable, no matter how
# or from where this script was launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHECKPOINT_DIR, OUTPUT_DIR, NUM_CLASSES, DEVICE
from dataset import SegmentationDataset
from model import SimpleUNet
from preprocessing import CLASS_COLORS


def class_ids_to_rgb(class_id_mask: np.ndarray) -> np.ndarray:
    """Turn a (H, W) class-id mask back into a colored (H, W, 3) image,
    using the same colors from class_dict.csv, so a human can look at it."""
    height, width = class_id_mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_index, (name, color) in enumerate(CLASS_COLORS):
        rgb_mask[class_id_mask == class_index] = color

    return rgb_mask


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = SimpleUNet(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    test_dataset = SegmentationDataset(split="test", augment=False)

    with torch.no_grad():
        for i in range(len(test_dataset)):
            image_tensor, _ = test_dataset[i]
            input_batch = image_tensor.unsqueeze(0).to(DEVICE)  # add batch dimension

            logits = model(input_batch)
            predicted_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

            rgb_mask = class_ids_to_rgb(predicted_mask)
            image_id = test_dataset.rows.iloc[i]["image_id"]

            out_path = os.path.join(OUTPUT_DIR, f"{image_id}_predicted_mask.png")
            Image.fromarray(rgb_mask).save(out_path)
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()

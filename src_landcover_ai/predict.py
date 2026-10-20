import os
import sys
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHECKPOINT_DIR, OUTPUT_DIR, NUM_CLASSES, DEVICE, CLASS_COLORS,
    BATCH_SIZE, NUM_WORKERS, CLASS_WEIGHTS, LOG_DIR, TEST_METRICS_CSV,
)
from dataset import SegmentationDataset
from model import SimpleUNet
from losses import get_loss_function
from metrics import pixel_accuracy, mean_iou
from preprocessing import resize_image, normalize_image


def class_ids_to_rgb(class_id_mask: np.ndarray) -> np.ndarray:
    """Turn a (H, W) class-id mask back into a colored (H, W, 3) image,
    using the hand-picked colors in config.py, so a human can look at it."""
    height, width = class_id_mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_index, (name, color) in enumerate(CLASS_COLORS):
        rgb_mask[class_id_mask == class_index] = color

    return rgb_mask


def load_model():
    model = SimpleUNet(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def predict_test_split(model):
    """Runs inference on every tile in test.txt, saves each predicted
    mask as an image, AND (since LandCover.ai's test split ships real
    ground-truth masks, unlike DeepGlobe's) computes loss/accuracy/IoU
    against those real masks - this is the truest measure of how the
    model performs, since test tiles were never seen during training or
    used to pick the best checkpoint."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    test_dataset = SegmentationDataset(split="test", augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )

    # same weighted loss used in training, so this loss number is
    # directly comparable to the train/valid loss you already have
    loss_fn = get_loss_function(class_weights=CLASS_WEIGHTS, device=DEVICE)

    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    num_batches = 0
    tile_index = 0  # test_loader is shuffle=False, so this tracks position
                     # in test_dataset.names in lockstep with each batch

    with torch.no_grad():
        for images, masks in test_loader:
            images_dev = images.to(DEVICE, non_blocking=True)
            masks_dev = masks.to(DEVICE, non_blocking=True)

            logits = model(images_dev)
            loss = loss_fn(logits, masks_dev)
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            total_acc += pixel_accuracy(predictions, masks_dev)
            total_iou += mean_iou(predictions, masks_dev, NUM_CLASSES)
            num_batches += 1

            predictions_cpu = predictions.cpu().numpy()
            for i in range(predictions_cpu.shape[0]):
                name = test_dataset.names[tile_index]
                rgb_mask = class_ids_to_rgb(predictions_cpu[i])
                out_path = os.path.join(OUTPUT_DIR, f"{name}_predicted_mask.png")
                Image.fromarray(rgb_mask).save(out_path)
                tile_index += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_iou = total_iou / num_batches

    print(f"saved {tile_index} predicted masks to {OUTPUT_DIR}")
    print(f"test set  loss={avg_loss:.4f}  acc={avg_acc:.4f}  iou={avg_iou:.4f}")

    with open(TEST_METRICS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss", "accuracy", "mean_iou"])
        writer.writerow([avg_loss, avg_acc, avg_iou])
    print(f"saved {TEST_METRICS_CSV}")


def predict_single_image(model, image_path):
    """Runs inference on one arbitrary image (e.g. something from the
    internet, not part of the LandCover.ai dataset at all) - no ground
    truth exists for this, so only the predicted mask is saved, no
    metrics."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image = resize_image(image)
    image_array = normalize_image(np.array(image))
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
    input_batch = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_batch)
        predicted_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

    rgb_mask = class_ids_to_rgb(predicted_mask)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_predicted_mask.png")
    Image.fromarray(rgb_mask).save(out_path)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                         help="path to a single image to run prediction on, "
                              "instead of the full test.txt split")
    args = parser.parse_args()

    model = load_model()

    if args.image:
        predict_single_image(model, args.image)
    else:
        predict_test_split(model)


if __name__ == "__main__":
    main()
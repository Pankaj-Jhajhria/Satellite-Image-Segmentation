import os
import sys
import csv
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_CLASSES, CHECKPOINT_DIR,
    DEVICE, CLASS_WEIGHTS, NUM_WORKERS, USE_AMP, LOG_DIR, HISTORY_CSV,
)
from dataset import SegmentationDataset
from model import SimpleUNet
from losses import get_loss_function
from metrics import pixel_accuracy, mean_iou

# lets cuDNN pick the fastest convolution algorithms for our fixed
# 256x256 input size - only helps on CUDA, harmless on CPU
torch.backends.cudnn.benchmark = True

AMP_ENABLED = USE_AMP and DEVICE == "cuda"


def run_one_epoch(model, loader, loss_fn, optimizer=None, scaler=None):
    """If optimizer is given, this trains one epoch.
    If optimizer is None, this only evaluates (used for validation)."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0

    with torch.set_grad_enabled(is_training):
        for images, masks in loader:
            # non_blocking=True lets the CPU->GPU copy overlap with other
            # work instead of stalling the training loop on every batch
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=AMP_ENABLED):
                logits = model(images)
                loss = loss_fn(logits, masks)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                if AMP_ENABLED:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            total_acc += pixel_accuracy(predictions, masks)
            total_iou += mean_iou(predictions, masks, NUM_CLASSES)

    num_batches = len(loader)
    return total_loss / num_batches, total_acc / num_batches, total_iou / num_batches


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # fresh history file every run - so old and new runs never get mixed
    # together in one chart by accident
    with open(HISTORY_CSV, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "train_iou", "val_loss", "val_acc", "val_iou"]
        )

    print(f"device: {DEVICE}   mixed precision: {AMP_ENABLED}")

    train_dataset = SegmentationDataset(split="train", augment=True)
    valid_dataset = SegmentationDataset(split="valid", augment=False)

    print(f"train tiles: {len(train_dataset)}   valid tiles: {len(valid_dataset)}")

    pin_memory = DEVICE == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
    )

    model = SimpleUNet(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = get_loss_function(class_weights=CLASS_WEIGHTS, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    if CLASS_WEIGHTS is None:
        print("CLASS_WEIGHTS is None - training with plain unweighted loss. "
              "Run compute_class_weights.py first if you want to correct "
              "for woodland dominance.")

    best_val_iou = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_iou = run_one_epoch(model, train_loader, loss_fn, optimizer, scaler)
        val_loss, val_acc, val_iou = run_one_epoch(model, valid_loader, loss_fn, optimizer=None)

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"  train  loss={train_loss:.4f}  acc={train_acc:.4f}  iou={train_iou:.4f}")
        print(f"  valid  loss={val_loss:.4f}  acc={val_acc:.4f}  iou={val_iou:.4f}")

        # append immediately, not buffered until the end - so even if
        # training is interrupted partway through, everything up to the
        # last finished epoch is already saved to disk
        with open(HISTORY_CSV, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, train_iou, val_loss, val_acc, val_iou])

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved new best model (val_iou={val_iou:.4f})")


if __name__ == "__main__":
    main()

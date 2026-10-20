import os
import sys
import torch
from torch.utils.data import DataLoader

# make sure this file's own folder (src/) is importable, no matter how
# or from where this script was launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_CLASSES, CHECKPOINT_DIR, DEVICE
from dataset import SegmentationDataset
from model import SimpleUNet
from losses import get_loss_function
from metrics import pixel_accuracy, mean_iou


def run_one_epoch(model, loader, loss_fn, optimizer=None):
    """If optimizer is given, this trains one epoch.
    If optimizer is None, this only evaluates (used for validation)."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0

    with torch.set_grad_enabled(is_training):
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            logits = model(images)
            loss = loss_fn(logits, masks)

            if is_training:
                optimizer.zero_grad()
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

    train_dataset = SegmentationDataset(split="train", augment=True)
    valid_dataset = SegmentationDataset(split="valid", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleUNet(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_iou = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_iou = run_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc, val_iou = run_one_epoch(model, valid_loader, loss_fn, optimizer=None)

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"  train  loss={train_loss:.4f}  acc={train_acc:.4f}  iou={train_iou:.4f}")
        print(f"  valid  loss={val_loss:.4f}  acc={val_acc:.4f}  iou={val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved new best model (val_iou={val_iou:.4f})")


if __name__ == "__main__":
    main()

"""Reads logs_landcover/history.csv (written automatically by train.py)
and saves a 3-panel chart: loss, accuracy, and IoU, train vs valid, over
epochs.

Run from the project root, any time after (or during) a training run:
    python src_landcover_model3/plot_history.py
"""
import os
import sys
import csv
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HISTORY_CSV, LOG_DIR

epochs, train_loss, train_acc, train_iou = [], [], [], []
val_loss, val_acc, val_iou = [], [], []

with open(HISTORY_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        train_acc.append(float(row["train_acc"]))
        train_iou.append(float(row["train_iou"]))
        val_loss.append(float(row["val_loss"]))
        val_acc.append(float(row["val_acc"]))
        val_iou.append(float(row["val_iou"]))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].plot(epochs, train_loss, label="train")
axes[0].plot(epochs, val_loss, label="valid")
axes[0].set_title("Loss")
axes[0].set_xlabel("epoch")
axes[0].legend()

axes[1].plot(epochs, train_acc, label="train")
axes[1].plot(epochs, val_acc, label="valid")
axes[1].set_title("Pixel accuracy")
axes[1].set_xlabel("epoch")
axes[1].legend()

axes[2].plot(epochs, train_iou, label="train")
axes[2].plot(epochs, val_iou, label="valid")
axes[2].set_title("Mean IoU")
axes[2].set_xlabel("epoch")
axes[2].legend()

plt.tight_layout()
out_path = os.path.join(LOG_DIR, "training_history.png")
plt.savefig(out_path)
print(f"saved {out_path}")

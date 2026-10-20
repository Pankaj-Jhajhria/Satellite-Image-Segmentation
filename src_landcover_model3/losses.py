import torch
import torch.nn as nn


def get_loss_function(class_weights=None, device="cpu"):
    """CrossEntropyLoss compares the model's per-pixel logits against
    the true class-id mask, and applies softmax internally.
    Never apply softmax yourself before this - it would break the math.

    class_weights: optional list of per-class weights, e.g. the output of
    compute_class_weights.py. Without this, LandCover.ai's heavy woodland
    dominance will very likely push the model toward the same kind of
    single-color collapse seen on DeepGlobe - just green here instead of
    yellow/blue."""
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=weight_tensor)
    return nn.CrossEntropyLoss()

import torch.nn as nn


def get_loss_function():
    """CrossEntropyLoss compares the model's per-pixel logits against
    the true class-id mask, and applies softmax internally.
    Never apply softmax yourself before this - it would break the math."""
    return nn.CrossEntropyLoss()

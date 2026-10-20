def pixel_accuracy(predictions, targets):
    """Fraction of pixels the model got exactly right."""
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total


def mean_iou(predictions, targets, num_classes):
    """Average IoU across every class that actually shows up in this batch."""
    ious = []
    for class_id in range(num_classes):
        pred_mask = predictions == class_id
        true_mask = targets == class_id

        intersection = (pred_mask & true_mask).sum().item()
        union = (pred_mask | true_mask).sum().item()

        if union == 0:
            continue  # this class isn't present in the batch, skip it

        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)

import torch

def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.float().clone()
    target = target.float()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice

def pixel_accuracy(prediction, target):
    if prediction.dtype != torch.bool:
        pred_labels = (prediction > 0).float()
    else:
        pred_labels = prediction.float()

    target_labels = target.float()

    correct_pixels = torch.sum(pred_labels == target_labels)
    total_pixels = torch.numel(target_labels)

    accuracy = correct_pixels / total_pixels

    return accuracy.item()


def iou_score(prediction, target, epsilon=1e-7):
    pred_labels = (prediction > 0).float()
    target_labels = target.float()

    intersection = torch.sum(pred_labels * target_labels)
    union = torch.sum(pred_labels) + torch.sum(target_labels) - intersection

    iou = (intersection + epsilon) / (union + epsilon)

    return iou.item()
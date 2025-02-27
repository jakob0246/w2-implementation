import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import warnings


def prepare_output_and_target(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    return output, target


def label_int2index(array):
    array = array.cpu()
    result = np.zeros((len(array), 6))

    result[array == 0] = np.array([1, 0, 0, 0, 0, 0])
    result[array == 1] = np.array([0, 1, 0, 0, 0, 0])
    result[array == 2] = np.array([0, 0, 1, 0, 0, 0])
    result[array == 3] = np.array([0, 0, 0, 1, 0, 0])
    result[array == 4] = np.array([0, 0, 0, 0, 1, 0])
    result[array == 5] = np.array([0, 0, 0, 0, 0, 1])

    result = torch.tensor(result)

    return result


def compute_mean_iou(output, target):
    output = torch.argmax(output, dim=1)

    output = output.flatten()
    target = target.flatten()

    output = label_int2index(output)
    target = label_int2index(target)

    intersection = torch.sum(output * target, dim=0)
    union = torch.sum(target, dim=0) + torch.sum(output, dim=0) - intersection
    ious = (intersection + .0000001) / (union + .0000001)

    mean_iou = torch.mean(ious)
    return mean_iou


def compute_balanced_accuracy_mc(output, target):
    output = torch.argmax(output, dim=1)

    output = output.flatten().cpu()
    target = target.flatten().cpu()

    # Ignore UserWarning regarding `y_pred` index not in `y_true`
    warnings.filterwarnings("ignore")
    bal_accuracy = balanced_accuracy_score(output, target)
    warnings.filterwarnings("default")

    return bal_accuracy


def compute_balanced_accuracy(output, target):
    tp = true_positives(output, target)
    tn = true_negatives(output, target)
    fp = false_positives(output, target)
    fn = false_negatives(output, target)

    sensitivity = tp / (tp + fn + .0000001)
    specificity = tn / (tn + fp + .0000001)
    balanced_accuracy = (sensitivity + specificity) / 2

    return balanced_accuracy


def true_positives(output, target):
    output, target = prepare_output_and_target(output, target)
    correct = torch.sum(output * target)
    return correct


def true_negatives(output, target):
    output, target = prepare_output_and_target(output, target)
    output = (output == 0)
    target = (target == 0)
    correct = torch.sum(output * target)
    return correct


def false_positives(output, target):
    output, target = prepare_output_and_target(output, target)
    output = (output == 1)
    target = (target == 0)
    correct = torch.sum(output * target)
    return correct


def false_negatives(output, target):
    output, target = prepare_output_and_target(output, target)
    output = (output == 0)
    target = (target == 1)
    correct = torch.sum(output * target)
    return correct

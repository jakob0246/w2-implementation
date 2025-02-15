import warnings
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def prepare_output_and_target(output, target, ignore_index):
    output = torch.argmax(output, dim=1).flatten()
    target = torch.argmax(target, dim=1).flatten()

    if ignore_index is not None:
        no_ignore = target.ne(ignore_index).cuda()
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)

    return output, target


def label_int2index(array, num_classes):
    array = array.cpu()
    result = np.zeros((len(array), num_classes))

    for i in range(num_classes):
        label_index_array = np.zeros(num_classes)
        label_index_array[i] = 1
        result[array == i] = label_index_array

    result = torch.tensor(result)

    return result


def compute_mean_iou(output, target, num_classes, ignore_index):
    """ Returns the IoU for binary segmentation problems and the mIoU for multiclass ones. """

    output, target = prepare_output_and_target(output, target, ignore_index)

    output = label_int2index(output, num_classes)
    target = label_int2index(target, num_classes)

    intersection = torch.sum(output * target, dim=0)
    union = torch.sum(target, dim=0) + torch.sum(output, dim=0) - intersection
    ious = (intersection + .0000001) / (union + .0000001)

    # `iou` can be the IoU or mIoU here, depending on a binary or multiclass problem
    iou = ious[1] if num_classes == 2 else torch.mean(ious)

    return iou


def compute_balanced_accuracy_mc(output, target):
    output = torch.argmax(output, dim=1)

    output = output.flatten().cpu()
    target = target.flatten().cpu()

    # Ignore UserWarning regarding `y_pred` index not in `y_true`
    warnings.filterwarnings("ignore")
    bal_accuracy = balanced_accuracy_score(output, target)
    warnings.filterwarnings("default")

    return bal_accuracy


def compute_balanced_accuracy(output, target, ignore_index):
    tp = true_positives(output, target, ignore_index)
    tn = true_negatives(output, target, ignore_index)
    fp = false_positives(output, target, ignore_index)
    fn = false_negatives(output, target, ignore_index)

    sensitivity = tp / (tp + fn + .0000001)
    specificity = tn / (tn + fp + .0000001)
    balanced_accuracy = (sensitivity + specificity) / 2

    return balanced_accuracy


def true_positives(output, target, ignore_index):
    output, target = prepare_output_and_target(output, target, ignore_index)
    correct = torch.sum(output * target)
    return correct


def true_negatives(output, target, ignore_index):
    output, target = prepare_output_and_target(output, target, ignore_index)
    output = (output == 0)
    target = (target == 0)
    correct = torch.sum(output * target)
    return correct


def false_positives(output, target, ignore_index):
    output, target = prepare_output_and_target(output, target, ignore_index)
    output = (output == 1)
    target = (target == 0)
    correct = torch.sum(output * target)
    return correct


def false_negatives(output, target, ignore_index):
    output, target = prepare_output_and_target(output, target, ignore_index)
    output = (output == 0)
    target = (target == 1)
    correct = torch.sum(output * target)
    return correct

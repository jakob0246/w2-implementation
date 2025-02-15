import torch


def prepare_output_and_target(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)

    return output, target


def compute_iou(output, target):
    output, target = prepare_output_and_target(output, target)
    intersection = torch.sum(output * target)
    union = torch.sum(target) + torch.sum(output) - intersection
    iou = (intersection + .0000001) / (union + .0000001)

    if iou != iou:
        print("failed, replacing with 0")
        iou = torch.tensor(0).float()

    return iou


def compute_accuracy(output, target, device):
    output, target = prepare_output_and_target(output, target, device)
    correct = torch.sum(output.eq(target))
    return correct.float() / len(target)


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

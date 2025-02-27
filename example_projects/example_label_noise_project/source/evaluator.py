import torch
import numpy as np
from scipy.special import softmax

from metrics import compute_mean_iou, compute_balanced_accuracy_mc


def evaluate(network, eval_loader, criterion, use_seed=False):
    network = network.eval()
    network = network.cuda()

    previous_random_state = torch.random.get_rng_state()
    if use_seed:
        torch.manual_seed(1)

    intermediate_loss = 0
    intermediate_miou = 0
    intermediate_bal_acc = 0
    count = 0
    outputs = []
    with torch.no_grad():
        for (images, labels) in eval_loader:
            output = network(images.cuda())

            intermediate_loss += criterion(output, labels.long().cuda()).cpu() if criterion is not None else np.nan
            intermediate_miou += compute_mean_iou(output, labels.cuda()).cpu()
            intermediate_bal_acc += compute_balanced_accuracy_mc(output, labels.cuda())
            count += 1

            outputs.append(softmax(output.cpu(), axis=1))

    if use_seed:
        # Reset seed to be non-deterministically again
        torch.random.set_rng_state(previous_random_state)

    outputs = np.concatenate(outputs)

    metrics = {"loss": intermediate_loss / count, "miou": intermediate_miou / count, "accuracy": intermediate_bal_acc / count}

    return metrics, outputs

import torch
import numpy as np


def evaluate(network, eval_loader, use_seed=False, eval_mode=True):
    if eval_mode:
        network = network.eval()
    network = network.cuda()

    previous_random_state = torch.random.get_rng_state()
    if use_seed:
        torch.manual_seed(1)

    outputs = []
    with torch.no_grad():
        for images in eval_loader:
            output = network(images.cuda())
            outputs.append(output.cpu())

    if use_seed:
        # Reset seed to be non-deterministically again
        torch.random.set_rng_state(previous_random_state)

    outputs = np.concatenate(outputs)

    return outputs

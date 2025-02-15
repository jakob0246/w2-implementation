import numpy as np
import torch


def normalize(input_t):
    """
    Normalize every input channel to mean 0 and variance 1 for each input

    :param input_t: (torch.tensor or np.array) of shape (.., W, H)
    :return: input_normalized (torch.tensor)
    """
    input_as_array = np.asarray(input_t)

    means = np.nanmean(input_as_array, axis=(-2, -1), keepdims=True)
    stds = np.nanstd(input_as_array, axis=(-2, -1), keepdims=True)

    input_normalized = (input_as_array - means) / stds
    input_normalized = torch.from_numpy(input_normalized)

    return input_normalized


class InputNormalization:
    """Normalize the input image."""

    def __init__(self, means=None, stds=None):
        self.means = np.asarray(means).reshape((-1, 1, 1)) if means is not None else None
        self.stds = np.asarray(stds).reshape((-1, 1, 1)) if stds is not None else None

    def __call__(self, sample):
        # Normalize input
        img = np.asarray(sample['input'])
        means = img.mean(axis=(1, 2), keepdims=True) if self.means is None else self.means
        stds = img.std(axis=(1, 2), keepdims=True) if self.stds is None else self.stds
        img = (img - means) / stds
        sample['input'] = torch.tensor(img, dtype=torch.float32)

        return sample
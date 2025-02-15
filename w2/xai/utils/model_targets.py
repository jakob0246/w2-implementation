# modified from https://github.com/jacobgil/pytorch-grad-cam


import numpy as np
import torch


class Classification:
    def __init__(self, category: int):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


class SemanticSegmentation:
    """ Gets a binary spatial mask and a category,
        And return the sum of the category scores,
        of the pixels in the mask.
        see: Vinogradova et al. https://arxiv.org/abs/2002.11434
    """

    def __init__(self,
                 category: int,
                 mask: np.ndarray or None,
                 use_cuda: bool = torch.cuda.is_available()):
        self.category = category
        self.mask = torch.from_numpy(mask) if mask is not None else 1
        self.use_cuda = use_cuda
        if self.use_cuda and mask is not None:
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

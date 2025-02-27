import torch
import numpy as np
from scipy.special import softmax

from .utils.evaluator import evaluate
from .utils.utils import InMemoryDataset


def estimate_uncertainty(model, images, preprocess, collate_fn):
    """
    Applies test time data augmentation uncertainty estimation to given images.
    The approach applies horizontal-flip, vertical-flip, 180-rotations, multiplications and gaussian noise as
    augmentations.

    Parameters
    ----------
    model : torch.nn.Module
        A pre-trained convolutional network model that serves as the basis for the training of the student models.
    images : tensor
        A pytorch tensor of shape (#Images, #Channels, N, M).
    preprocess : callable
        A function being in the form of `(tensor) -> (tensor)`.
    collate_fn : callable
        Collate function for evaluating the model.
        Should be empty most of the time for standard use-cases.

    Returns
    -------
    model_output : array
        An array of model output logits of shape (#Images, #Classes, N, M).
    """

    eval_dataset = InMemoryDataset(images, preprocess)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=1,
                                              collate_fn=collate_fn, pin_memory=True)
    model_output = evaluate(model, eval_loader, use_seed=True)
    model_output = softmax(np.array(model_output), axis=1)

    return model_output

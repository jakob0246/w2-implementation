import copy
import torch
from tqdm import tqdm

from .utils.evaluator import evaluate
from .utils.utils import InMemoryDataset


def estimate_uncertainty(model, images, preprocess, num_samples, noise_factor, collate_fn):
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
    num_samples : int
        The number of sampling passes or rather evaluation passes of the model.
    noise_factor : float
        The amount of gaussian noise to be added to the weights per sampling pass.

    Returns
    -------
    model_outputs : list
        A list of length #Samples of model output logits of shape (#Images, #Classes, N, M).
    """

    eval_dataset = InMemoryDataset(images, preprocess)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=1,
                                              collate_fn=collate_fn, pin_memory=True)

    print()

    model_outputs = []
    for _ in tqdm(range(num_samples), desc=f"Running {num_samples} weight noise passes", unit="samples",
                  bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        model_clone = copy.deepcopy(model)
        # Inspired by source: https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829/2
        with torch.no_grad():
            for parameter in model_clone.parameters():
                parameter.add_(noise_factor * torch.randn(parameter.size()).cuda())

        model_output = evaluate(model_clone, eval_loader, use_seed=True)
        model_outputs.append(model_output)

    return model_outputs

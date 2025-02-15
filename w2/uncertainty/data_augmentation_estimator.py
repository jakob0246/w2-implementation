import torch
from tqdm import tqdm

from .utils.evaluator import evaluate
from .utils.utils import InMemoryDataset


def estimate_uncertainty(model, images, preprocess, postprocess, augmentations, noise_factor, collate_fn):
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
    postprocess : callable
        Should have the form `(tensor) -> (tensor)`.
    augmentations : list
        A list of augmentations to be applied.
    noise_factor : float
        The amount of gaussian noise to be applied.
    collate_fn : callable
        Collate function for evaluating the model.
        Should be empty most of the time for standard use-cases.

    Returns
    -------
    model_outputs : list
        A list of length #Samples of model output logits of shape (#Images, #Classes, N, M).
    """

    print()

    model_outputs = []
    for transform in tqdm(augmentations,
                          desc=f"Running {len(augmentations)} (+1) data augmentation passes", unit="samples",
                          bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        augmented_images = transform.augment_image(torch.stack(images))

        eval_dataset = InMemoryDataset(augmented_images, preprocess)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=1,
                                                  collate_fn=collate_fn, pin_memory=True)

        model_output = evaluate(model, eval_loader, use_seed=True)

        if postprocess is not None:
            model_output = postprocess(torch.from_numpy(model_output))
        else:
            model_output = torch.from_numpy(model_output)
        model_output = transform.deaugment_mask(model_output)

        model_outputs.append(model_output)

    # Add model outputs of gaussian noise augmentation
    augmented_images = gaussian_noise_augmentation(torch.stack(images), noise_factor)
    eval_dataset = InMemoryDataset(augmented_images, preprocess)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=1,
                                              collate_fn=collate_fn, pin_memory=True)
    model_output = evaluate(model, eval_loader, use_seed=True)
    if postprocess is not None:
        model_output = postprocess(torch.from_numpy(model_output))
    else:
        model_output = torch.from_numpy(model_output)
    model_outputs.append(model_output)

    return model_outputs


def gaussian_noise_augmentation(images, factor):
    # Add gaussian noise
    images += factor * torch.normal(0, float(torch.max(images)), size=images.shape)
    return images

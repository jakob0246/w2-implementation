import torch
import numpy as np
from scipy.special import softmax
from scipy.special import expit as sigmoid
from scipy.stats import entropy


def std_deviation_uncertainty_metric(outputs):
    """
    Takes all sampling output logits and combines them to one uncertainty map with help of
    utilizing the standard deviation.
    As a final step, standard deviation maps are normalized.

    Parameters
    ----------
    outputs : list
        List of size number of samples, containing the output logits of each sample.
        With that, list elements are numpy arrays with the shape (#Images, #Classes, N, M).

    Returns
    -------
    std_deviation_maps : tensor
        A standard deviation uncertainty map for each image, resulting in a pytorch tensor of shape (#Images, N, M).
    """

    outputs = softmax(np.array(outputs), axis=2)
    outputs = outputs[:, :, 1, :, :]
    # outputs = np.clip(outputs, 0, 1)
    std_deviation_maps = np.std(outputs, axis=0)
    # Normalize, since standard deviations have a small value range
    std_deviation_maps = std_deviation_maps / np.max(std_deviation_maps)
    std_deviation_maps = torch.tensor(std_deviation_maps)
    return std_deviation_maps


def std_deviation_uncertainty_metric_refined(outputs, pos_class=0):
    """
    Takes all sampling output logits and combines them to one uncertainty map with help of
    utilizing the standard deviation.
    As a final step, standard deviation maps are normalized.

    Parameters
    ----------
    outputs : list
        List of size number of samples, containing the output logits of each sample.
        With that, list elements are numpy arrays with the shape (#Images, #Classes, N, M).

    Returns
    -------
    std_deviation_maps : tensor
        A standard deviation uncertainty map for each image, resulting in a pytorch tensor of shape (#Images, N, M).
    """

    outputs = np.array(outputs)
    num_classes = outputs.shape[2]

    if num_classes == 1:
        outputs = sigmoid(np.array(outputs))[:, :, 0, :, :]
    else:
        outputs = softmax(np.array(outputs), axis=2)[:, :, pos_class, :, :]
    # if num_classes == 2:
        # outputs = outputs[:, :, 1, :, :]
    std_deviation_maps = np.std(outputs, axis=0)
    #if num_classes != 1:
        #std_deviation_maps = np.mean(std_deviation_maps, axis=1)
    # Normalize, since standard deviations have a small value range
    std_deviation_maps = np.array([_normalize(std_deviation_map) for std_deviation_map in std_deviation_maps])
    std_deviation_maps = torch.tensor(std_deviation_maps)
    return std_deviation_maps


def softmax_uncertainty_metric(output, pos_class=0):
    """
    Takes the output logits of one evaluation pass and focuses on the positive class, e.g. the flood class.
    As a final step, maps are normalized.

    Parameters
    ----------
    output : array
        Array containing the output logits of the model of shape (#Images, #Classes, N, M).
    pos_class : int
        Positive class of the labels, e.g., a flood class.

    Returns
    -------
    uncertainty_map : tensor
        A standard deviation uncertainty map for each image, resulting in a pytorch tensor of shape (#Images, N, M).
    """

    output = np.array(output)[:, pos_class, :, :]
    output[output > 0.5] = 1 - output[output > 0.5]
    uncertainty_map = np.array([_normalize(single_output) for single_output in output])
    uncertainty_map = torch.tensor(uncertainty_map)
    return uncertainty_map


def counter_max_prob_metric(outputs, pos_class):
    """TODO: explain this metric!"""
    outputs = np.mean(np.array(outputs), axis=0)
    num_classes = outputs.shape[1]
    if num_classes == 1:
        outputs = sigmoid(outputs)[:, 0, :, :]
    else:
        outputs = softmax(outputs, axis=1)[:, pos_class, :, :]
    outputs[outputs > 0.5] = 1 - outputs[outputs > 0.5]
    # Normalize
    uncertainty_map = np.array([_normalize(output) for output in outputs])
    uncertainty_map = torch.tensor(uncertainty_map)
    return uncertainty_map


def two_highest_prob_metric(outputs):
    outputs = np.mean(outputs, axis=0)
    outputs = softmax(outputs, axis=1)
    outputs = np.sort(outputs, axis=1)
    uncertainty_maps = 1 - (outputs[:, 1, :, :] - outputs[:, 0, :, :])
    # Normalize
    uncertainty_maps = np.array([_normalize(uncertainty_map) for uncertainty_map in uncertainty_maps])
    uncertainty_maps = torch.tensor(uncertainty_maps)
    return uncertainty_maps


def true_class_prob_metric(outputs, labels, ignore_index):
    """ Only works for the binary class case right now! """

    print(labels.shape)

    outputs = np.mean(np.array(outputs), axis=0)
    outputs = softmax(outputs, axis=1)

    outputs_class0 = outputs[:, 0, :, :].copy()
    outputs_class1 = outputs[:, 1, :, :].copy()
    outputs_true = outputs_class0.copy()
    outputs_true[labels == 1] = outputs_class1[labels == 1]

    uncertainty_maps = np.max(outputs, axis=1) - outputs_true
    # Normalize
    uncertainty_maps = np.array([_normalize(uncertainty_map) for uncertainty_map in uncertainty_maps])

    uncertainty_maps[labels == ignore_index] = np.nan

    uncertainty_maps = torch.tensor(uncertainty_maps)
    return uncertainty_maps


def entropy_metric(outputs):
    outputs = np.mean(np.array(outputs), axis=0)
    outputs = softmax(np.array(outputs), axis=1)
    outputs = entropy(outputs, axis=1)
    # Normalize
    uncertainty_maps = np.array([_normalize(output) for output in outputs])
    uncertainty_maps = torch.tensor(uncertainty_maps)
    return uncertainty_maps


def _normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

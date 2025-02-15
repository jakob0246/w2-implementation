import torch
import numpy as np
from scipy.special import softmax
from sklearn.calibration import calibration_curve

from .evaluator import evaluate
from .utils import (InMemoryDataset, plot_confidence_miou, plot_calibration, plot_calibration_multiclass,
                    label_int2index)


def _normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def generate_binary_calibration(model, val_data, sampling_outputs, methods, preprocess, postprocess,
                                ensemble_temperature, ignore_index, collate_fn, results_directory):
    """ Generate a plot of calibration curves. """

    # Build validation loader
    val_images = [val_image_and_label[0] for val_image_and_label in val_data]
    val_dataset = InMemoryDataset(val_images, preprocess)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1,
                                             collate_fn=collate_fn, pin_memory=True)

    # Evaluate teacher model
    val_outputs = evaluate(model, val_loader, use_seed=True)
    val_outputs = torch.tensor(val_outputs)
    if postprocess is not None:
        val_outputs = postprocess(val_outputs)
    val_outputs = softmax(val_outputs / ensemble_temperature, axis=1)
    val_outputs = val_outputs[:, 1, :, :].flatten()

    val_labels = np.array([np.array(val_image_and_label[1]) for val_image_and_label in val_data]).flatten()

    # Remove invalid annotated labels
    if ignore_index is not None:
        invalid_indices = (val_labels == ignore_index)
        val_labels_filtered = np.delete(val_labels, invalid_indices)
        val_outputs = np.delete(val_outputs, invalid_indices)
        output_y, output_x = calibration_curve(val_labels_filtered, val_outputs, n_bins=15)
    else:
        output_y, output_x = calibration_curve(val_labels, val_outputs, n_bins=15)

    # Remove invalid annotated labels
    invalid_indices = None
    if ignore_index is not None:
        invalid_indices = (val_labels == ignore_index)
        val_labels = np.delete(val_labels, invalid_indices)

    uncertainties_x = []
    uncertainties_y = []
    for i, method in enumerate(methods):
        if method == "softmax":
            continue

        sampling_output = sampling_outputs[i]

        # Aggregate sampling outputs
        sampling_output = np.array([np.array(postprocess(torch.tensor(single_output))
                                        if postprocess is not None and method != "data-augmentation" else single_output)
                                    for single_output in sampling_output])
        sampling_output = np.mean(sampling_output, axis=0)

        if ensemble_temperature == 1.0:
            sampling_output = softmax(sampling_output, axis=1)[:, 1, :, :].flatten()
        else:
            sampling_output = sampling_output[:, 1, :, :].flatten()
            sampling_output = np.clip(sampling_output, 0, 1)

        # Remove invalid annotated labels
        if ignore_index is not None:
            sampling_output = np.delete(sampling_output, invalid_indices)

        uncertainty_y, uncertainty_x = calibration_curve(val_labels, sampling_output, n_bins=15)
        uncertainties_x.append(uncertainty_x)
        uncertainties_y.append(uncertainty_y)

    plot_calibration(output_x, output_y, uncertainties_x, uncertainties_y, methods, results_directory)


def generate_multiclass_calibration(model, val_data, sampling_output, method, preprocess, postprocess, num_classes,
                                    ensemble_temperature, ignore_index, collate_fn, results_directory, confidence=False):
    """ Generate a plot of calibration curves. """

    # Build validation loader
    val_images = [val_image_and_label[0] for val_image_and_label in val_data]
    val_dataset = InMemoryDataset(val_images, preprocess)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1,
                                             collate_fn=collate_fn, pin_memory=True)

    # Evaluate teacher model
    val_outputs = evaluate(model, val_loader, use_seed=True)
    val_outputs = torch.tensor(val_outputs)
    if postprocess is not None:
        val_outputs = postprocess(val_outputs)
    val_outputs = np.array(val_outputs)
    val_outputs = softmax(val_outputs / ensemble_temperature, axis=1)

    # Aggregate sampling outputs
    sampling_output = np.array([np.array(postprocess(torch.tensor(single_output))
                                    if postprocess is not None and method != "data-augmentation" else single_output)
                                for single_output in sampling_output])
    sampling_output = np.mean(sampling_output, axis=0)
    sampling_output = softmax(sampling_output, axis=1)

    val_labels = np.array([np.array(val_image_and_label[1]) for val_image_and_label in val_data])

    # Remove invalid annotated labels
    if ignore_index is not None:
        invalid_indices = (val_labels == ignore_index)
        val_outputs = np.delete(val_outputs, invalid_indices)
        val_labels = np.delete(val_labels, invalid_indices)
        sampling_output = np.delete(sampling_output, invalid_indices)

    if not confidence:
        outputs_x, outputs_y = [], []
        uncertainties_x, uncertainties_y = [], []
        for i in range(num_classes):
            output_x, output_y = determine_multiclass_bins_classwise(val_outputs, val_labels, pos_class=i)
            uncertainty_x, uncertainty_y = determine_multiclass_bins_classwise(sampling_output, val_labels,
                                                                               pos_class=i)
            outputs_x.append(output_x), outputs_y.append(output_y)
            uncertainties_x.append(uncertainty_x), uncertainties_y.append(uncertainty_y)
        plot_calibration_multiclass(outputs_x, outputs_y, uncertainties_x, uncertainties_y, num_classes,
                                    results_directory)
    else:
        output_x, output_y = determine_multiclass_confidence_miou(val_outputs, val_labels)
        uncertainty_x, uncertainty_y = determine_multiclass_confidence_miou(sampling_output, val_labels)
        plot_confidence_miou(output_x, output_y, uncertainty_x, uncertainty_y, results_directory)


def determine_multiclass_bins_classwise(predictions, labels, pos_class=0, num_bins=10):
    predictions_pos_class = predictions[:, pos_class, :, :].flatten()
    labels = labels.flatten()

    x = []
    y = []
    bin_bounds = np.linspace(0.0, 1.0, num_bins + 1)
    for i, bin_lower_bound in enumerate(bin_bounds):
        if bin_lower_bound == 1.0:
            break

        bin_upper_bound = bin_bounds[i + 1]
        bin_indices = (bin_lower_bound <= predictions_pos_class) & (predictions_pos_class < bin_upper_bound)

        labels_bin = labels[bin_indices]
        predictions_pos_class_bin = predictions_pos_class[bin_indices]

        bin_y = np.sum(labels_bin == pos_class) / len(predictions_pos_class_bin)

        y.append(bin_y)
        x.append(np.mean(predictions_pos_class_bin))

    return x, y


def determine_multiclass_confidence_miou(predictions, labels):
    predictions_argmax = np.argmax(predictions, axis=1).flatten()
    predictions_max = np.max(predictions, axis=1).flatten()
    labels = labels.flatten()

    y = []
    taus = np.linspace(0, 0.9, 10)
    for i, tau in enumerate(taus):
        pred_max_ge_tau_indices = predictions_max >= tau
        if np.sum(pred_max_ge_tau_indices) != 0:
            predictions_argmax_subset = predictions_argmax[pred_max_ge_tau_indices]
            labels_subset = labels[pred_max_ge_tau_indices]

            predictions_argmax_subset = label_int2index(predictions_argmax_subset, 6)
            labels_subset = label_int2index(labels_subset, 6)

            intersection = np.sum(predictions_argmax_subset * labels_subset, axis=0)
            union = np.sum(predictions_argmax_subset, axis=0) + np.sum(labels_subset, axis=0) - intersection
            ious = (intersection + .0000001) / (union + .0000001)
            miou = np.mean(ious)

            y.append(miou)

    x = list(taus)[:(len(y))]

    return x, y

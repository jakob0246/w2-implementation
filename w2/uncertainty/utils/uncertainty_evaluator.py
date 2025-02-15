import torch
import numpy as np
import copy
from scipy.special import softmax
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import get_cmap

from .evaluator import evaluate
from .utils import InMemoryDataset


def evaluate_uncertainty_vs_error_rate(model, val_data, uncertainties, methods, preprocess, postprocess, ignore_index,
                                       collate_fn, save_dir, num_bins=15):
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
    val_outputs = softmax(val_outputs, axis=1)
    val_outputs = np.argmax(val_outputs, axis=1).flatten()

    val_labels = np.array([np.array(val_image_and_label[1]) for val_image_and_label in val_data]).flatten()

    invalid_indices = []
    if ignore_index is not None:
        invalid_indices = (val_labels == ignore_index)
        val_labels = np.delete(val_labels, invalid_indices)
        val_outputs = np.delete(val_outputs, invalid_indices)

    figure = go.Figure()

    for i, uncertainty in enumerate(uncertainties):
        # Remove invalid annotated labels
        if ignore_index is not None:
            uncertainty = np.delete(uncertainty.flatten(), invalid_indices)

        y = []
        bin_bounds = np.linspace(0.0, 1.0, num_bins + 1)
        for j, bin_lower_bound in enumerate(bin_bounds):
            if bin_lower_bound == 1.0:
                break

            bin_upper_bound = bin_bounds[j + 1]
            bin_indices = (bin_lower_bound <= uncertainty) & (uncertainty < bin_upper_bound)

            val_labels_bin = val_labels[bin_indices]
            val_outputs_bin = val_outputs[bin_indices]

            y.append(1 - np.sum(val_outputs_bin == val_labels_bin) / len(val_labels_bin))

        x = bin_bounds[:-1] + 1 / num_bins / 2

        title = methods[i].title().replace("-", " ")
        if methods[i] == "dropout":
            title = "Monte Carlo Dropout"
        if methods[i] == "weight-noise":
            title = "Parameter Noise"

        figure.add_trace(go.Scatter(x=x, y=y, name=title))

    figure.update_layout(
        height=550,
        width=700,
        font_family="DejaVu Sans",
        font_size=11,
        plot_bgcolor='#E4F4FA',
        legend=dict(
            x=0.07,  # x=0.63
            y=0.93,  # y=0.07
            traceorder="normal",
            font=dict(
                size=10
            )
        )
    )

    figure["layout"]["xaxis"]["title"] = "Uncertainty Score"
    figure["layout"]["yaxis"]["title"] = "Error Rate"

    filename = f"uncertainty_vs_error"
    figure.write_image(f"{save_dir}/{filename}.pdf", scale=3.0)


def evaluate_metrics(uncertainties, metrics, save_dir):
    # TODO: one colorbar -> standardize between heatmaps

    custom_cmap = copy.copy(get_cmap("viridis"))
    custom_cmap.set_bad((0.2666, 0.0039, 0.3294))

    fig, axs = plt.subplots(nrows=len(metrics) - 1, ncols=len(metrics) - 1, figsize=(13, 15.5))
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            if i <= j:
                if i != len(metrics) - 1 and j != len(metrics) - 1 and i != j:
                    axs[i, j].axis("off")
                continue

            uncertainty_x = np.array(uncertainties[i].flatten())
            uncertainty_y = np.array(uncertainties[j].flatten())
            filter_indices = (np.isnan(uncertainty_x) | np.isnan(uncertainty_y))
            uncertainty_x = np.delete(uncertainty_x, filter_indices)
            uncertainty_y = np.delete(uncertainty_y, filter_indices)

            _, _, _, im = axs[i - 1, j].hist2d(uncertainty_x, uncertainty_y, bins=20, norm=colors.LogNorm(),
                                               cmap=custom_cmap)
            fig.colorbar(im, ax=axs[i - 1, j], orientation="horizontal", pad=0.12)
            axs[i - 1, j].plot([0, 1], [0, 1], color="red")

    titles = ["$\\sigma$", "$1 - \\max_i p(z_i|x)$",
              "$\\max_i p(z_i|x) - \\max_i^2 p(z_i|x)$",
              "$\\max_i p(z_i|x) - p(y|x)$", "$H(p(z))$"]

    labelpadding = 40
    axs[0, 0].set_ylabel(titles[1])
    axs[1, 0].set_ylabel(titles[2])
    axs[2, 0].set_ylabel(titles[3])
    axs[3, 0].set_ylabel(titles[4])
    axs[3, 0].set_xlabel(titles[0], labelpad=labelpadding)
    axs[3, 1].set_xlabel(titles[1], labelpad=labelpadding)
    axs[3, 2].set_xlabel(titles[2], labelpad=labelpadding)
    axs[3, 3].set_xlabel(titles[3], labelpad=labelpadding)

    plt.grid(False)
    plt.subplots_adjust(wspace=0.3, hspace=0.1)

    save_directory = save_dir
    filename = "metric_correlations"
    plt.savefig(f"{save_directory}/{filename}.pdf")

def evaluate_methods(uncertainties, methods, save_dir):
    # TODO: one colorbar -> standardize between heatmaps

    custom_cmap = copy.copy(get_cmap("viridis"))
    custom_cmap.set_bad((0.2666, 0.0039, 0.3294))

    fig, axs = plt.subplots(nrows=len(methods) - 1, ncols=len(methods) - 1, figsize=(13, 15.5))
    for i in range(len(methods)):
        for j in range(len(methods)):
            if i <= j:
                if i != len(methods) - 1 and j != len(methods) - 1 and i != j:
                    axs[i, j].axis("off")
                continue

            uncertainty_x = np.array(uncertainties[i].flatten())
            uncertainty_y = np.array(uncertainties[j].flatten())
            filter_indices = (np.isnan(uncertainty_x) | np.isnan(uncertainty_y))
            uncertainty_x = np.delete(uncertainty_x, filter_indices)
            uncertainty_y = np.delete(uncertainty_y, filter_indices)

            _, _, _, im = axs[i - 1, j].hist2d(uncertainty_x, uncertainty_y, bins=20, norm=colors.LogNorm(),
                                               cmap=custom_cmap)
            fig.colorbar(im, ax=axs[i - 1, j], orientation="horizontal", pad=0.12)
            axs[i - 1, j].plot([0, 1], [0, 1], color="red")

    titles = ["Student Ensemble", "Monte Carlo Dropout", "Data Augmentation", "Weight Noise", "Softmax"]

    labelpadding = 40
    axs[0, 0].set_ylabel(titles[1])
    axs[1, 0].set_ylabel(titles[2])
    axs[2, 0].set_ylabel(titles[3])
    axs[3, 0].set_ylabel(titles[4])
    axs[3, 0].set_xlabel(titles[0], labelpad=labelpadding)
    axs[3, 1].set_xlabel(titles[1], labelpad=labelpadding)
    axs[3, 2].set_xlabel(titles[2], labelpad=labelpadding)
    axs[3, 3].set_xlabel(titles[3], labelpad=labelpadding)

    plt.grid(False)
    plt.subplots_adjust(wspace=0.3, hspace=0.1)

    save_directory = save_dir
    filename = "method_correlations"
    plt.savefig(f"{save_directory}/{filename}.pdf")

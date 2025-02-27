import csv
import os
import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

from utils import int2rgb


def generate_convergence_plots(train_stats, val_stats, save_dir=os.path.join("..", "results")):
    figure = make_subplots(rows=3, cols=1)

    figure.add_trace(
        go.Scatter(
            y=val_stats["losses"],
            name="Validation Loss",
            line=go.scatter.Line(color="#025464"),
            legendgroup="1"
        ),
        row=1, col=1
    )
    figure.add_trace(
        go.Scatter(
            y=train_stats["losses"],
            name="Train Loss",
            line=go.scatter.Line(color="#025464"),
            opacity=0.5,
            marker=go.scatter.Marker(symbol="x"),
            legendgroup="1"
        ),
        row=1, col=1
    )

    figure.add_trace(
        go.Scatter(
            y=val_stats["mious"],
            name="Validation mIoU",
            line=go.scatter.Line(color="#E57C23"),
            legendgroup="2"
        ),
        row=2, col=1
    )
    figure.add_trace(
        go.Scatter(
            y=train_stats["mious"],
            name="Train mIoU",
            line=go.scatter.Line(color="#E57C23"),
            opacity=0.5,
            marker=go.scatter.Marker(symbol="x"),
            legendgroup="2"
        ),
        row=2, col=1
    )

    figure.add_trace(
        go.Scatter(
            y=val_stats["accuracies"],
            name="Validation Bal. Accuracy",
            line=go.scatter.Line(color="#917FB3"),
            legendgroup="3"
        ),
        row=3, col=1
    )
    figure.add_trace(
        go.Scatter(
            y=train_stats["accuracies"],
            name="Train Bal. Accuracy",
            line=go.scatter.Line(color="#917FB3"),
            opacity=0.5,
            marker=go.scatter.Marker(symbol="x"),
            legendgroup="3"
        ),
        row=3, col=1
    )

    figure.update_layout(
        title_text="Training Convergence",
        height=800,
        width=1200,
        yaxis1_title="Loss",
        yaxis2_title="mIoU",
        yaxis3_title="Balanced Accuracy",
        xaxis3_title="Epoch",
        legend_tracegroupgap=190
    )

    filename = "train_convergence_plot"
    figure.write_image(f"{save_dir}/{filename}.pdf", scale=3.0)


def generate_example_uncertainty_maps(data, outputs, uncertainty_maps, save_dir=os.path.join("..", "results"),
                                      is_val=False):
    images = [np.array(image_and_label[0]) for image_and_label in data]
    labels = [int2rgb(image_and_label[1]) for image_and_label in data]

    labels_prediction = np.argmax(outputs, axis=1)
    labels_prediction = [int2rgb(label) for label in labels_prediction]

    # Convert uncertainty maps to heat maps
    rgb_uncertainty_maps = []
    for uncertainty_map in list(uncertainty_maps):
        uncertainty_map = np.array(uncertainty_map)
        uncertainty_map = (uncertainty_map * 255).astype(np.uint8)
        uncertainty_map = uncertainty_map[:, :, np.newaxis]
        rgb_uncertainty_map = cv2.applyColorMap(uncertainty_map, cv2.COLORMAP_JET)
        rgb_uncertainty_map = cv2.cvtColor(rgb_uncertainty_map, cv2.COLOR_RGB2BGR)
        rgb_uncertainty_maps.append(rgb_uncertainty_map.astype(np.float32))

    figure = make_subplots(rows=4, cols=len(data))

    [figure.add_trace(go.Image(z=image), row=1, col=i + 1) for i, image in enumerate(images)]
    [figure.add_trace(go.Image(z=label), row=2, col=i + 1) for i, label in enumerate(labels)]
    [figure.add_trace(go.Image(z=label), row=3, col=i + 1) for i, label in enumerate(labels_prediction)]
    [figure.add_trace(go.Image(z=rgb_uncertainty_map), row=4, col=i + 1) for i, rgb_uncertainty_map in enumerate(rgb_uncertainty_maps)]

    figure.update_layout(
        title_text=f"Uncertainty Maps for Example {'Validation' if is_val else 'Test'} Images",
        height=800,
        width=1200
    )

    figure["layout"]["yaxis"]["title"] = "Images"
    figure["layout"]["yaxis6"]["title"] = "Reference Labels"
    figure["layout"]["yaxis11"]["title"] = "Predictions"
    figure["layout"]["yaxis16"]["title"] = "Uncertainty"

    filename = f"example_{'validation' if is_val else 'test'}_uncertainty_maps"
    figure.write_image(f"{save_dir}/{filename}.pdf", scale=3.0)


def write_teacher_statistics(noise_combination, train_metrics, val_metrics, test_metrics, results_path):
    path_to_file = os.path.join(results_path, "teacher_statistics.csv")
    file_already_exists = os.path.isfile(path_to_file)

    with open(path_to_file, "a") as file:
        writer = csv.writer(file)

        if not file_already_exists:
            writer.writerow(["seed", "pixeltype", "noise", "selection-method", "relabel-method", "train-mIoU",
                             "val-mIoU", "test-mIoU", "train-balanced-acc", "val-balanced-acc", "test-balanced-acc"])

        writer.writerow([noise_combination[0], noise_combination[1], noise_combination[2], noise_combination[3],
                         noise_combination[4], train_metrics["miou"].item(), val_metrics["miou"].item(),
                         test_metrics["miou"].item(), train_metrics["accuracy"], val_metrics["accuracy"],
                         test_metrics["accuracy"]])

        file.close()


def generate_uncertainty_images(data, outputs, uncertainty_maps):
    images = [np.array(image_and_label[0]) for image_and_label in data]
    labels = [int2rgb(image_and_label[1]) for image_and_label in data]

    labels_prediction = np.argmax(outputs, axis=1)
    labels_prediction = [int2rgb(label) for label in labels_prediction]

    # Convert uncertainty maps to heat maps
    rgb_uncertainty_maps = []
    for uncertainty_map in list(uncertainty_maps):
        uncertainty_map = np.array(uncertainty_map)
        uncertainty_map = (uncertainty_map * 255).astype(np.uint8)
        uncertainty_map = uncertainty_map[:, :, np.newaxis]
        rgb_uncertainty_map = cv2.applyColorMap(uncertainty_map, cv2.COLORMAP_JET)  # cv2.COLORMAP_PLASMA
        rgb_uncertainty_map = cv2.cvtColor(rgb_uncertainty_map, cv2.COLOR_RGB2BGR)
        rgb_uncertainty_maps.append(rgb_uncertainty_map.astype(np.float32))

    save_directory = "../results/images"
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    for i in range(len(images)):
        rgb_image = Image.fromarray(images[i].astype("uint8"))
        rgb_image.save(f"{save_directory}/{i}_image.jpg")
        rgb_label_ground_truth = Image.fromarray(labels[i].astype("uint8"))
        rgb_label_ground_truth.save(f"{save_directory}/{i}_label.jpg")
        rgb_label_prediction = Image.fromarray(labels_prediction[i].astype("uint8"))
        rgb_label_prediction.save(f"{save_directory}/{i}_prediction.jpg")
        rgb_uncertainty_map = Image.fromarray(rgb_uncertainty_maps[i].astype("uint8"))
        rgb_uncertainty_map.save(f"{save_directory}/{i}_uncertainty.jpg")


def generate_uncertainty_images_noisy(images, labels, outputs, uncertainty_maps, save_directory):
    images = [np.array(image) for image in images]
    labels = [int2rgb(label) for label in labels]

    labels_prediction = np.argmax(outputs, axis=1)
    labels_prediction = [int2rgb(label) for label in labels_prediction]

    # Convert uncertainty maps to heat maps
    rgb_uncertainty_maps = []
    for uncertainty_map in list(uncertainty_maps):
        uncertainty_map = np.array(uncertainty_map)
        uncertainty_map = (uncertainty_map * 255).astype(np.uint8)
        uncertainty_map = uncertainty_map[:, :, np.newaxis]
        rgb_uncertainty_map = cv2.applyColorMap(uncertainty_map, cv2.COLORMAP_JET)  # cv2.COLORMAP_PLASMA
        rgb_uncertainty_map = cv2.cvtColor(rgb_uncertainty_map, cv2.COLOR_RGB2BGR)
        rgb_uncertainty_maps.append(rgb_uncertainty_map.astype(np.float32))

    save_directory = f"{save_directory}/images"
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    for i in range(len(images)):
        rgb_image = Image.fromarray(images[i].astype("uint8"))
        rgb_image.save(f"{save_directory}/{i}_image.jpg")
        rgb_label_ground_truth = Image.fromarray(labels[i].astype("uint8"))
        rgb_label_ground_truth.save(f"{save_directory}/{i}_label.jpg")
        rgb_label_prediction = Image.fromarray(labels_prediction[i].astype("uint8"))
        rgb_label_prediction.save(f"{save_directory}/{i}_prediction.jpg")
        rgb_uncertainty_map = Image.fromarray(rgb_uncertainty_maps[i].astype("uint8"))
        rgb_uncertainty_map.save(f"{save_directory}/{i}_uncertainty.jpg")

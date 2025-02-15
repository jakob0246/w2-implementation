from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np
from PIL import Image
import cv2

from utils import stitch_2_by_2_patches


def generate_convergence_plots(train_stats, val_stats):
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
            y=val_stats["ious"],
            name="Validation IoU",
            line=go.scatter.Line(color="#E57C23"),
            legendgroup="2"
        ),
        row=2, col=1
    )
    figure.add_trace(
        go.Scatter(
            y=train_stats["ious"],
            name="Train IoU",
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
        yaxis2_title="IoU",
        yaxis3_title="Accuracy",
        xaxis3_title="Epoch",
        legend_tracegroupgap=190
    )

    save_directory = "../results"
    filename = "train_convergence_plot"
    figure.write_image(f"{save_directory}/{filename}.pdf", scale=3.0)


def apply_min_max_contrast_stretching(channel):
    min_value = np.min(channel)
    max_value = np.max(channel)
    stretched_channel = ((channel - min_value) / ((max_value - min_value) if max_value - min_value != 0 else 0)) * 255
    return stretched_channel


def create_images_labels_predictions(outputs, data):
    blue_color = np.array([116, 211, 250])

    # Convert SAR images to RGB & add ground truth labels
    rgb_images = []
    rgb_labels_ground_truth = []
    for (sar_image, label) in data:
        sar_image, label = sar_image.numpy(), label.numpy()

        # Enhance image depiction: clipping and min / max stretching
        vv_channel = np.clip(sar_image[0] * 255, 128, 230)
        vh_channel = np.clip(sar_image[1] * 255, 77, 200)
        vv_channel = apply_min_max_contrast_stretching(vv_channel)
        vh_channel = apply_min_max_contrast_stretching(vh_channel)

        blue_channel = np.zeros_like(vv_channel)
        rgb_image = np.dstack((vv_channel, vh_channel, blue_channel))
        rgb_images.append(rgb_image)

        rgb_label_ground_truth = np.zeros_like(rgb_image)
        rgb_label_ground_truth[label.squeeze() == 1] = blue_color
        rgb_label_ground_truth[label.squeeze() == 255] = np.array([255, 255, 255])
        rgb_labels_ground_truth.append(rgb_label_ground_truth)

    rgb_labels_prediction = []
    for output in list(outputs):
        rgb_label_prediction = np.zeros_like(rgb_images[0])
        rgb_label_prediction[output == 1] = blue_color
        rgb_labels_prediction.append(rgb_label_prediction)

    return rgb_images, rgb_labels_ground_truth, rgb_labels_prediction


def generate_example_uncertainty_maps(data, outputs, uncertainty_maps, filenames, is_val=False):
    rgb_images, rgb_labels_ground_truth, rgb_labels_prediction = create_images_labels_predictions(outputs, data)

    # Convert uncertainty maps to heat maps
    rgb_uncertainty_maps = []
    for uncertainty_map in list(uncertainty_maps):
        uncertainty_map = np.array(uncertainty_map)
        uncertainty_map = (uncertainty_map * 255).astype(np.uint8)
        uncertainty_map = uncertainty_map[:, :, np.newaxis]
        rgb_uncertainty_map = cv2.applyColorMap(uncertainty_map, cv2.COLORMAP_JET)  # cv2.COLORMAP_PLASMA
        rgb_uncertainty_map = cv2.cvtColor(rgb_uncertainty_map, cv2.COLOR_RGB2BGR)
        rgb_uncertainty_maps.append(rgb_uncertainty_map.astype(np.float32))

    figure = make_subplots(rows=4, cols=len(data))

    [figure.add_trace(go.Image(z=rgb_image), row=1, col=i + 1) for i, rgb_image in enumerate(rgb_images)]
    [figure.add_trace(go.Image(z=rgb_label_ground_truth), row=2, col=i + 1) for i, rgb_label_ground_truth in enumerate(rgb_labels_ground_truth)]
    [figure.add_trace(go.Image(z=rgb_label_prediction), row=3, col=i + 1) for i, rgb_label_prediction in enumerate(rgb_labels_prediction)]
    [figure.add_trace(go.Image(z=rgb_uncertainty_map), row=4, col=i + 1) for i, rgb_uncertainty_map in enumerate(rgb_uncertainty_maps)]

    figure.update_layout(
        title_text=f"Uncertainty Maps for Example {'Validation' if is_val else 'Test'} Images",
        font_size=20,
        height=1600,
        width=2400,
        xaxis16_title=filenames[0],
        xaxis17_title=filenames[1],
        xaxis18_title=filenames[2],
        xaxis19_title=filenames[3],
        xaxis20_title=filenames[4]
    )

    figure["layout"]["yaxis"]["title"] = "SAR Images"
    figure["layout"]["yaxis6"]["title"] = "Reference Flood Data"
    figure["layout"]["yaxis11"]["title"] = "Predictions"
    figure["layout"]["yaxis16"]["title"] = "Uncertainty"

    save_directory = "../results"
    filename = f"example_{'validation' if is_val else 'test'}_uncertainty_maps"
    figure.write_image(f"{save_directory}/{filename}.pdf", scale=1.0)


def generate_uncertainty_images(data, outputs, uncertainty_maps, filenames, is_val=False):
    rgb_images, rgb_labels_ground_truth, rgb_labels_prediction = create_images_labels_predictions(outputs, data)

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

    for i, file in enumerate(filenames):
        rgb_image = Image.fromarray(rgb_images[i].astype("uint8"))
        rgb_image.save(f"{save_directory}/{file}_image.jpg")
        rgb_label_ground_truth = Image.fromarray(rgb_labels_ground_truth[i].astype("uint8"))
        rgb_label_ground_truth.save(f"{save_directory}/{file}_label.jpg")
        rgb_label_prediction = Image.fromarray(rgb_labels_prediction[i].astype("uint8"))
        rgb_label_prediction.save(f"{save_directory}/{file}_prediction.jpg")
        rgb_uncertainty_map = Image.fromarray(rgb_uncertainty_maps[i].astype("uint8"))
        rgb_uncertainty_map.save(f"{save_directory}/{file}_uncertainty.jpg")


def generate_example_teacher_softmax_maps(outputs, data, filenames, is_val):
    rgb_images, rgb_labels_ground_truth, rgb_labels_prediction = create_images_labels_predictions(outputs, data)

    outputs = outputs[:, 1, :, :]
    outputs = stitch_2_by_2_patches(outputs)

    # Convert softmax maps to heat maps
    rgb_softmax_maps = []
    for output in outputs:
        output = (output * 255).astype(np.uint8)
        output = output[:, :, np.newaxis]
        rgb_softmax_map = cv2.applyColorMap(output, cv2.COLORMAP_JET)
        rgb_softmax_map = cv2.cvtColor(rgb_softmax_map, cv2.COLOR_RGB2BGR)
        rgb_softmax_maps.append(rgb_softmax_map.astype(np.float32))

    figure = make_subplots(rows=4, cols=len(data))

    [figure.add_trace(go.Image(z=rgb_image), row=1, col=i + 1) for i, rgb_image in enumerate(rgb_images)]
    [figure.add_trace(go.Image(z=rgb_label_ground_truth), row=2, col=i + 1) for i, rgb_label_ground_truth in enumerate(rgb_labels_ground_truth)]
    [figure.add_trace(go.Image(z=rgb_label_prediction), row=3, col=i + 1) for i, rgb_label_prediction in enumerate(rgb_labels_prediction)]
    [figure.add_trace(go.Image(z=rgb_uncertainty_map), row=4, col=i + 1) for i, rgb_uncertainty_map in enumerate(rgb_softmax_maps)]

    figure.update_layout(
        title_text=f"Teacher Softmax Maps for Example {'Validation' if is_val else 'Test'} Images",
        height=800,
        width=1200,
        xaxis16_title=filenames[0],
        xaxis17_title=filenames[1],
        xaxis18_title=filenames[2],
        xaxis19_title=filenames[3],
        xaxis20_title=filenames[4]
    )

    figure["layout"]["yaxis"]["title"] = "SAR Images"
    figure["layout"]["yaxis6"]["title"] = "Reference Flood Data"
    figure["layout"]["yaxis11"]["title"] = "Predictions"
    figure["layout"]["yaxis16"]["title"] = "Teacher Softmax Maps"

    save_directory = "../results"
    filename = f"example_{'validation' if is_val else 'test'}_teacher_softmax_maps"
    figure.write_image(f"{save_directory}/{filename}.pdf", scale=1.0)

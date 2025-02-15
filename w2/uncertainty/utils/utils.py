import os
import csv
from tqdm import tqdm
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .evaluator import evaluate
from .metrics import compute_mean_iou, compute_balanced_accuracy_mc


class InMemoryDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)


def label_int2index(array, num_classes):
    result = np.zeros((len(array), num_classes))

    for i in range(num_classes):
        label_index_array = np.zeros(num_classes)
        label_index_array[i] = 1
        result[array == i] = label_index_array

    return result


# TODO
def write_student_ensemble_statistics(teacher_model, val_images, student_model_outputs, preprocess, num_classes,
                                      collate_fn, model_saves_directory):
    val_dataset = InMemoryDataset(val_images, preprocess)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1,
                                             collate_fn=collate_fn, pin_memory=True)
    teacher_model_output = evaluate(teacher_model, val_loader)

    mean_squared_error = torch.nn.MSELoss()

    mses = []
    mious = []
    accuracies = []
    for student_model_output in student_model_outputs:
        teacher_model_output = torch.tensor(teacher_model_output)
        student_model_output = torch.tensor(student_model_output)

        mses.append(mean_squared_error(teacher_model_output, student_model_output).item())
        mious.append(compute_mean_iou(teacher_model_output, student_model_output, num_classes, None).item())

        student_model_output_argmax = torch.argmax(student_model_output, dim=1)
        accuracies.append(compute_balanced_accuracy_mc(teacher_model_output, student_model_output_argmax).item())

    # TODO: Use better results-path parameter
    results_path = os.path.join(os.path.sep, *model_saves_directory.split("/")[:-2], "results")

    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    path_to_file = os.path.join(results_path, "student_statistics.csv")

    with open(path_to_file, "w") as file:
        writer = csv.writer(file)

        writer.writerow(["student", "val-mse", "val-mIoU", "val-balanced-acc"])

        for i in range(len(student_model_outputs)):
            writer.writerow([i, mses[i], mious[i], accuracies[i]])

        file.close()


def plot_calibration(output_x, output_y, uncertainties_x, uncertainties_y, methods, results_directory):
    methods_copy = methods.copy()

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=output_x, y=output_y, name="Teacher Output", line=go.scatter.Line(dash="4px")))

    if "softmax" in methods_copy:
        methods_copy.remove("softmax")

    for i, method in enumerate(methods_copy):
        title = method.title().replace("-", " ")
        if method == "dropout":
            title = "Monte Carlo Dropout"
        if method == "weight-noise":
            title = "Parameter Noise"
        figure.add_trace(go.Scatter(x=uncertainties_x[i], y=uncertainties_y[i], name=title))

    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=go.scatter.Line(color="Gray"), opacity=0.65, mode="lines",
                                showlegend=False))

    figure.update_layout(
        height=700,
        width=700,
        font_family="DejaVu Sans",
        font_size=10,
        plot_bgcolor='#E4F4FA',
        legend=dict(
            x=0.07,
            y=0.93,
            traceorder="normal",
            font=dict(
                size=10
            )
        )
    )

    figure["layout"]["xaxis"]["title"] = "Mean Predicted Probability"
    figure["layout"]["yaxis"]["title"] = "Fraction of Positives"

    figure.update_xaxes(showgrid=True, gridwidth=2, gridcolor='White')
    figure.update_yaxes(showgrid=True, gridwidth=2, gridcolor='White')

    filename = f"calibration_curves"
    figure.write_image(f"{results_directory}/{filename}.pdf", scale=3.0)

    tqdm.write(f"\nWrote calibration curves to {results_directory}")


def plot_confidence_miou(output_x, output_y, uncertainty_x, uncertainty_y, results_directory):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=output_x, y=output_y, name="Teacher Output"))
    figure.add_trace(go.Scatter(x=uncertainty_x, y=uncertainty_y, name="Mean Ensemble Output"))

    figure.update_layout(
        title_text=f"Confidence mIoU Curves",
        shapes=[{"type": "line", "yref": "paper", "xref": "paper", "y0": 0, "y1": 1, "x0": 0, "x1": 1, "opacity": 0.5}]
    )

    figure["layout"]["xaxis"]["title"] = "$\\text{Confidence threshold } \\tau$"
    figure["layout"]["yaxis"]["title"] = "$\\text{mIoU on examples } p(y|x) \geq \\tau$"

    filename = f"calibration_curves"
    figure.write_image(f"{results_directory}/{filename}.pdf", scale=3.0)

    tqdm.write(f"\nWrote calibration curves to {results_directory}")


def plot_calibration_multiclass(outputs_x, outputs_y, uncertainties_x, uncertainties_y, num_classes, results_directory):
    """ TODO: Make independent of number of classes """

    figure = make_subplots(rows=3, cols=3,
                           subplot_titles=("class 0", "class 1", "class 2", "class 3", "class 4", "class 5", "mean"),
                           x_title="Mean Predicted Probability", y_title="Fraction of Positives")

    figure.add_trace(go.Scatter(x=[0.05, 1], y=[0.05, 1], line=go.scatter.Line(color="Gray"), opacity=0.65,
                                mode="lines", showlegend=False), row=1, col=1)
    figure.add_trace(go.Scatter(x=outputs_x[0], y=outputs_y[0], line=go.scatter.Line(color="#636EFA"),
                                name="Teacher Output"), row=1, col=1)
    figure.add_trace(go.Scatter(x=uncertainties_x[0], y=uncertainties_y[0], line=go.scatter.Line(color="#EF553B"),
                                name="Mean Ensemble Output"), row=1, col=1)

    figure.add_trace(go.Scatter(x=[0.05, 1], y=[0.05, 1], line=go.scatter.Line(color="Gray"), opacity=0.65,
                                mode="lines", showlegend=False), row=1, col=2)
    figure.add_trace(go.Scatter(x=outputs_x[1], y=outputs_y[1], line=go.scatter.Line(color="#636EFA"),
                                showlegend=False), row=1, col=2)
    figure.add_trace(go.Scatter(x=uncertainties_x[1], y=uncertainties_y[1], line=go.scatter.Line(color="#EF553B"),
                                showlegend=False), row=1, col=2)

    figure.add_trace(go.Scatter(x=[0.05, 1], y=[0.05, 1], line=go.scatter.Line(color="Gray"), opacity=0.65,
                                mode="lines", showlegend=False), row=1, col=3)
    figure.add_trace(go.Scatter(x=outputs_x[2], y=outputs_y[2], line=go.scatter.Line(color="#636EFA"),
                                showlegend=False), row=1, col=3)
    figure.add_trace(go.Scatter(x=uncertainties_x[2], y=uncertainties_y[2], line=go.scatter.Line(color="#EF553B"),
                                showlegend=False), row=1, col=3)

    figure.add_trace(go.Scatter(x=[0.05, 1], y=[0.05, 1], line=go.scatter.Line(color="Gray"), opacity=0.65,
                                mode="lines", showlegend=False), row=2, col=1)
    figure.add_trace(go.Scatter(x=outputs_x[3], y=outputs_y[3], line=go.scatter.Line(color="#636EFA"),
                                showlegend=False), row=2, col=1)
    figure.add_trace(go.Scatter(x=uncertainties_x[3], y=uncertainties_y[3], line=go.scatter.Line(color="#EF553B"),
                                showlegend=False), row=2, col=1)

    figure.add_trace(go.Scatter(x=[0.05, 1], y=[0.05, 1], line=go.scatter.Line(color="Gray"), opacity=0.65,
                                mode="lines", showlegend=False), row=2, col=2)
    figure.add_trace(go.Scatter(x=outputs_x[4], y=outputs_y[4], line=go.scatter.Line(color="#636EFA"),
                                showlegend=False), row=2, col=2)
    figure.add_trace(go.Scatter(x=uncertainties_x[4], y=uncertainties_y[4], line=go.scatter.Line(color="#EF553B"),
                                showlegend=False), row=2, col=2)

    figure.add_trace(go.Scatter(x=[0.05, 1], y=[0.05, 1], line=go.scatter.Line(color="Gray"), opacity=0.65,
                                mode="lines", showlegend=False), row=2, col=3)
    figure.add_trace(go.Scatter(x=outputs_x[5], y=outputs_y[5], line=go.scatter.Line(color="#636EFA"),
                                showlegend=False), row=2, col=3)
    figure.add_trace(go.Scatter(x=uncertainties_x[5], y=uncertainties_y[5], line=go.scatter.Line(color="#EF553B"),
                                showlegend=False), row=2, col=3)

    outputs_x_mean = np.nanmean(np.array(outputs_x), axis=0)
    outputs_y_mean = np.nanmean(np.array(outputs_y), axis=0)
    uncertainties_x_mean = np.nanmean(np.array(uncertainties_x), axis=0)
    uncertainties_y_mean = np.nanmean(np.array(uncertainties_y), axis=0)

    figure.add_trace(go.Scatter(x=[0.05, 1], y=[0.05, 1], line=go.scatter.Line(color="Gray"), opacity=0.65,
                                mode="lines", showlegend=False), row=3, col=1)
    figure.add_trace(go.Scatter(x=outputs_x_mean, y=outputs_y_mean, line=go.scatter.Line(color="#636EFA"),
                                showlegend=False), row=3, col=1)
    figure.add_trace(go.Scatter(x=uncertainties_x_mean, y=uncertainties_y_mean, line=go.scatter.Line(color="#EF553B"),
                                showlegend=False), row=3, col=1)

    figure.update_layout(
        title_text=f"Calibration Curves"
    )

    filename = f"calibration_curves"
    figure.write_image(f"{results_directory}/{filename}.pdf", scale=3.0)

    tqdm.write(f"\nWrote calibration curves to {results_directory}")

import os
import torch
import numpy as np

from .uncertainty import (student_ensemble_estimator, data_augmentation_estimator, softmax_estimator, dropout_estimator,
                          weight_noise_estimator)
from .uncertainty.utils.uncertainty_metrics import (std_deviation_uncertainty_metric_refined,
                                                    softmax_uncertainty_metric, counter_max_prob_metric,
                                                    two_highest_prob_metric, true_class_prob_metric, entropy_metric)
from .uncertainty.utils.calibration import generate_binary_calibration, generate_multiclass_calibration
from .uncertainty.utils.uncertainty_evaluator import (evaluate_uncertainty_vs_error_rate, evaluate_metrics,
                                                      evaluate_methods)


class UC:
    """
    Estimate the uncertainty with various methods on a pre-trained model and test images.

    Attributes
    ----------
    _available_methods : list
        Lists all available methods.
        Available methods should have the type string. When creating an `UC` object the method has to be included here.
    _available_modes : list
        Lists all available modes.
        Available modes should have the type string. A mode defines a specific machine learning subarea.
        Like with available methods, a specified mode has to be in the list.
    _default_params : dict
        A dictionary containing all default parameters for all uncertainty quantification methods.
    methods : list
        State a string of all uncertainty quantification methods that should be applied to test images.
    mode : str
        Define the mode out of all machine learning subareas.
    preprocess : callable
        A function being in the form of `(tensor) -> (tensor)`.
        Takes an image (pytorch tensor) of shape (#Channels, N, M) and returns a pytorch tensor of
        shape (#Images, #Channels, N, M), where #Images can be an arbitrary image augmentation multiplier.
        View the function like a modification of an image that returns the modified image again.
        The return value has an additional dimension, since the preprocessing step could include dividing the input
        image into multiple sub-images for example, or any other augmentation method.
    params : dict
        A dictionary containing all uncertainty estimation method parameters, including the default ones.
        Valid keys for each method are:
            - "student-ensemble"
                - "ensemble_train_params" (dict, see docs of
                    `student_ensemble_estimator.estimate_uncertainty() method`)
                - Optional: "num_students" (int), "num_epochs" (int), "num_classes" (int),
                    "student_model_saves_directory" (str), "ensemble_train_images" (tensor), "ensemble_train_preprocess"
                    (callable, ((tensor, tensor)) -> (tensor, tensor), first elements of the
                    input / output tuples are images of shape (#Channels, N, M) & second elements are labels of shape
                    (N, M))
            - "data-augmentation"
                - "augmentations" (tta.Compose) from https://github.com/qubvel/ttach, "noise_factor" (float)
            - "dropout"
                - "num_samples" (int)
            - "weight-noise"
                - "num_samples" (int), "noise_factor" (float)
    postprocess : callable, optional
        Postprocesses a set of tensors in the final step of applying the uncertainty method.
        Should have the form `(tensor) -> (tensor)`. The elements from the input and output tensors have the shape
        (#Images, #Channels, N, M). Note, that #Images for the input and output of the function could be different, e.g.
        when the function applies a reduce operation.
    collate_fn : callable, optional
        Collate function for evaluating the model.
        Should be empty most of the time for standard use-cases.
    """

    _available_methods = ["student-ensemble", "data-augmentation", "softmax", "dropout", "weight-noise"]
    _available_modes = ["segmentation"]
    _available_metrics = ["std", "counter_max", "second_max", "true_class", "entropy"]

    _default_params = {
        "student-ensemble": {
            "num_students": 10,
            "num_epochs": 100,
            "num_classes": 2,
            "ensemble_train_images": None,
            "ensemble_train_preprocess": None,
            "student_model_saves_directory": os.path.join("..", "model_saves", "students")
        },
        "data-augmentation": {
            "noise_factor": 0.2
        },
        "dropout": {
            "num_samples": 20
        },
        "weight-noise": {
            "num_samples": 100,
            "noise_factor": 0.005
        }
    }

    def __init__(self, methods, mode, model, preprocess, params, postprocess=None,
                 evaluation_collate=None):
        for method in methods:
            assert method in self._available_methods, "Passed method not available"
        assert mode in self._available_modes, "Passed mode not available"

        self.methods = methods
        self.mode = mode
        self.model = model
        self.preprocess = preprocess
        self.params = params
        self.postprocess = postprocess
        self.collate_fn = evaluation_collate

        self._fill_up_with_default_params()

    def predict(self, images, metric="std", pos_class=1, labels=None, label_preprocess=None, ignore_index=None):
        """
        Predict uncertainty quantification values for a given list of images.

        Parameters
        ----------
        images : tensor
            A pytorch tensor of shape (#Images, #Channels, N, M).
            The number of channels could for example be 2 for SAR images or 3 for optical RGB images.
        metric : str, optional
            The uncertainty quantification metric.
            Available metrics are: "std", "counter_max", "second_max", "true_class" and "entropy".
            The default is "std".
            Explanation for the metrics:
                - "std": Standard deviation of the model output logits.
                - "counter_max": 1 - probability of highest class.
                - "second_max": 1 - (probability of highest and second-highest probability).
                - "true_class": Highest probability - true class probability.
                - "entropy": Entropy of the model output distribution.
        labels : tensor, optional
            Reference labels to use if "true_class" metric should be applied.
            Otherwise, the parameter will be ignored.
        ignore_index : int, optional
            Index to be ignored for the labels.
            Only use if `true_class` metric should be applied. Otherwise, the parameter will be ignored.

        Returns
        -------
        uncertainty_maps : tensor
            A 4-dimensional pytorch tensor of shape (#Methods, #Images, N, M).
        """

        assert metric in self._available_metrics, "Passed metric not available"

        uncertainties = {}
        for method in self.methods:
            model_sampling_output = self._get_model_sampling_output(method, images)

            # Apply postprocessing function from package call
            if self.postprocess is not None and method != "data-augmentation":
                if method == "softmax":
                    model_sampling_output = self.postprocess(torch.tensor(model_sampling_output))
                else:
                    model_sampling_output = [np.array(self.postprocess(torch.tensor(output)))
                                             for output in model_sampling_output]

            print("\nApplying uncertainty metric ...")

            if method == "softmax":  # and metric == "counter_max":
                uncertainty = softmax_uncertainty_metric(model_sampling_output, pos_class)
            else:
                #if method == "softmax":
                    #uncertainty = model_sampling_output[:, pos_class, :, :]

                if metric == "std":
                    uncertainty = std_deviation_uncertainty_metric_refined(model_sampling_output, pos_class)
                elif metric == "counter_max":
                    uncertainty = counter_max_prob_metric(model_sampling_output, pos_class)
                elif metric == "second_max":
                    uncertainty = two_highest_prob_metric(model_sampling_output)
                elif metric == "true_class":
                    if method != "data-augmentation" and label_preprocess is not None:
                        uncertainty = true_class_prob_metric(model_sampling_output, labels,
                                                             ignore_index)
                    else:
                        uncertainty = true_class_prob_metric(model_sampling_output, labels, ignore_index)
                else:
                    uncertainty = entropy_metric(model_sampling_output)

            uncertainties[method] = uncertainty

        return uncertainties

    def generate_calibration_curves(self, num_classes, val_data, ignore_index=None, save_dir=os.path.join("..", "results")):
        """
        Generate a plot of calibration curves.
        The plot will include a curve of the teacher output softmax values and the uncertainty estimator.
        It will be written to ../results/

        Parameters
        ----------
        num_classes : int
            State the number of classes.
        val_data : list
            A list of tuples having the following format: `(tensor, tensor)` for images and labels from the shape
            (#Channels, N, M) and (N, M).
        ignore_index : int, optional
            Index to be ignored for generating the calibration curves.
        save_dir : string, optional
            Path were the plots of the calibration curves should be saved.
        """

        print("\nAnalyzing calibration ...")

        model_sampling_outputs = []
        for method in self.methods:
            val_images = [val_image_and_label[0] for val_image_and_label in val_data]
            model_sampling_output = self._get_model_sampling_output(method, val_images)
            model_sampling_outputs.append(model_sampling_output)

        if "student-ensemble" in self.methods:
            ensemble_temperature = self.params["student-ensemble"]["ensemble_train_params"]["temperature"]
        else:
            ensemble_temperature = 1.0

        if num_classes == 2:
            generate_binary_calibration(
                self.model,
                val_data,
                model_sampling_outputs,
                self.methods,
                self.preprocess,
                self.postprocess,
                ensemble_temperature,
                ignore_index,
                self.collate_fn,
                save_dir
            )
        else:
            generate_multiclass_calibration(
                self.model,
                val_data,
                model_sampling_output,
                method,
                self.preprocess,
                self.postprocess,
                ensemble_temperature,
                num_classes,
                ignore_index,
                self.collate_fn,
                save_dir
            )

    def evaluate_uncertainty_vs_error(self, val_data, metric="std", label_preprocess=None, ignore_index=None,
                                      save_dir=os.path.join("..", "results")):
        val_images = [val_image_and_label[0] for val_image_and_label in val_data]

        if metric == "true_class":
            val_labels = np.array([val_image_and_label[1] for val_image_and_label in val_data])
        else:
            val_labels = None

        uncertainties = self.predict(val_images, metric, val_labels, label_preprocess, ignore_index)

        evaluate_uncertainty_vs_error_rate(self.model, val_data, uncertainties, self.methods, self.preprocess,
                                           self.postprocess, ignore_index, self.collate_fn, save_dir)

    def evaluate_metric_correlation(self, val_data, label_preprocess=None, ignore_index=None,
                                    save_dir=os.path.join("..", "results")):
        """
        Evaluates the correlation of uncertainty scores for all available metrics for the first method in `methods`.
        """

        val_images = [val_image_and_label[0] for val_image_and_label in val_data]
        val_labels = np.array([val_image_and_label[1] for val_image_and_label in val_data])

        uncertainties = []
        for metric in self._available_metrics:
            if metric == "true_class":
                uncertainty = self.predict(val_images, metric, val_labels, label_preprocess, ignore_index)[0]
            else:
                uncertainty = self.predict(val_images, metric=metric)[0]
            uncertainties.append(uncertainty)

        evaluate_metrics(uncertainties, self._available_metrics, save_dir)

    def evaluate_method_correlation(self, val_data, save_dir=os.path.join("..", "results")):
        """
        Evaluates the correlation of uncertainty scores for all available methods with the standard deviation metric.
        """

        val_images = [val_image_and_label[0] for val_image_and_label in val_data]
        uncertainties = self.predict(val_images)
        evaluate_methods(uncertainties, self.methods, save_dir)

    def _get_model_sampling_output(self, method, images):
        """
        Get variance samples for uncertainty estimation.
        Each sample contains the logits of the model when applied on the `images`.

        Parameters
        ----------
        method : str
            State the uncertainty quantification method that should be utilized for the calibration curves comparison.
        images : tensor
            A pytorch tensor of shape (#Images, #Channels, N, M).
            The number of channels could for example be 2 for SAR images or 3 for optical RGB images.
        """

        model_sampling_output = None
        if method == "student-ensemble":
            model_sampling_output = student_ensemble_estimator.estimate_uncertainty(
                self.model,
                images,
                self.preprocess,
                self.postprocess,
                self.params[method]["num_students"],
                self.params[method]["num_epochs"],
                self.params[method]["num_classes"],
                self.params[method]["ensemble_train_images"],
                self.params[method]["ensemble_train_preprocess"],
                self.params[method]["ensemble_train_params"],
                self.collate_fn,
                self.params[method]["student_model_saves_directory"]
            )
        elif method == "data-augmentation":
            model_sampling_output = data_augmentation_estimator.estimate_uncertainty(
                self.model,
                images,
                self.preprocess,
                self.postprocess,
                self.params[method]["augmentations"],
                self.params[method]["noise_factor"],
                self.collate_fn
            )
        elif method == "softmax":
            model_sampling_output = softmax_estimator.estimate_uncertainty(
                self.model,
                images,
                self.preprocess,
                self.collate_fn
            )
        elif method == "dropout":
            model_sampling_output = dropout_estimator.estimate_uncertainty(
                self.model,
                images,
                self.preprocess,
                self.params[method]["num_samples"],
                self.collate_fn
            )
        elif method == "weight-noise":
            model_sampling_output = weight_noise_estimator.estimate_uncertainty(
                self.model,
                images,
                self.preprocess,
                self.params[method]["num_samples"],
                self.params[method]["noise_factor"],
                self.collate_fn
            )
        return model_sampling_output

    def _fill_up_with_default_params(self):
        """ Fills up the `params` attribute with missing `_default_params` parameters. """

        new_params = self.params.copy()
        for method in list(self.params.keys()):
            for parameter in list(self._default_params[method].keys()):
                if parameter not in list(self.params[method].keys()):
                    new_params[method][parameter] = self._default_params[method][parameter]
        self.params = new_params

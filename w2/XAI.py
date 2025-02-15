import time

import torch

from w2.xai.eigen_cam import EigenCAM
from w2.xai.grad_cam import GradCAM
from w2.xai.grad_cam_plusplus import GradCAMPlusPlus
from w2.xai.score_cam import ScoreCAM
from w2.xai.utils.model_targets import SemanticSegmentation, Classification


class XAI:
    """TODO: Write docstring here"""
    _available_methods = ['GradCAM', 'GradCAM++', 'ScoreCAM', 'EigenCAM']
    _available_modes = ['classification', 'segmentation', 'object_detection']
    _default_params = {
        "use_cuda": torch.cuda.is_available(),
        "eigen_smooth": False
    }

    def __init__(
            self,
            model,
            methods: list,
            mode: str,
            target_layers: list,
            preprocess=None,
            postprocess=None,
            params: dict = None
    ):
        for method in methods:
            assert method in self._available_methods, (f"Method {method} not available! "
                                                       f"Available methods: {self._available_methods}. Skipping...")
        assert mode in self._available_modes, f"Mode {mode} not available! Choose from {self._available_modes}."

        self.model = model
        self.methods = methods
        self.mode = mode
        self.target_layers = target_layers
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.params = self._reset_params(params)

    def explain(self, inp, class_num, mask, verbose=True):
        # define explainability targets
        if self.mode == "segmentation":
            targets = [SemanticSegmentation(class_num, mask, use_cuda=self.params["use_cuda"])]
        elif self.mode == "classification":
            if class_num is None:
                targets = None
            else:
                targets = [Classification(class_num)]
        elif self.mode == "object_detection":
            raise NotImplementedError("Object detection not implemented yet.")
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented. Choose from {self._available_modes}.")

        # run explainability methods
        smoothen_heatmaps = self.params["eigen_smooth"]
        cam_by_method = {}
        for method in self.methods:
            if method == "GradCAM":
                print("Running GradCAM...", end="") if verbose else None
                start = time.time()
                with GradCAM(model=self.model,
                             target_layers=self.target_layers,
                             use_cuda=self.params["use_cuda"]) as cam:
                    grayscale_cam = cam(input_tensor=inp, targets=targets, eigen_smooth=smoothen_heatmaps)
                cam_by_method[method] = grayscale_cam
                print(f" Done in {time.time() - start:.2f} seconds") if verbose else None
            if method == "GradCAM++":
                print("Running GradCAM++...", end="") if verbose else None
                start = time.time()
                with GradCAMPlusPlus(model=self.model,
                                     target_layers=self.target_layers,
                                     use_cuda=self.params["use_cuda"]) as cam:
                    grayscale_cam = cam(input_tensor=inp, targets=targets, eigen_smooth=smoothen_heatmaps)
                cam_by_method[method] = grayscale_cam
                print(f" Done in {time.time() - start:.2f} seconds") if verbose else None
            if method == "ScoreCAM":
                print("Running ScoreCAM...", end="") if verbose else None
                start = time.time()
                with ScoreCAM(model=self.model,
                              target_layers=self.target_layers,
                              use_cuda=self.params["use_cuda"]) as cam:
                    grayscale_cam = cam(input_tensor=inp, targets=targets, eigen_smooth=smoothen_heatmaps)
                cam_by_method[method] = grayscale_cam
                print(f" Done in {time.time() - start:.2f} seconds") if verbose else None
            if method == "EigenCAM":
                print("Running EigenCAM...", end="") if verbose else None
                start = time.time()
                with EigenCAM(model=self.model,
                              target_layers=self.target_layers,
                              use_cuda=self.params["use_cuda"]) as cam:
                    grayscale_cam = cam(input_tensor=inp, targets=targets, eigen_smooth=smoothen_heatmaps)
                cam_by_method[method] = grayscale_cam
                print(f" Done in {time.time() - start:.2f} seconds") if verbose else None

        return cam_by_method

    def _reset_params(self, params):
        new_params = self._default_params.copy()
        if params is None:
            return new_params
        for key, value in params.items():
            if key not in new_params.keys():
                raise KeyError(f"Parameter {key} not available and will be ignored.")
            new_params[key] = value
        return new_params



# modified from https://github.com/jacobgil/pytorch-grad-cam


import numpy as np
from w2.xai.base_cam import BaseCAM


class GradCAM(BaseCAM):
    name = "Grad-CAM"

    def __init__(self, model, target_layers, use_cuda=False):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))

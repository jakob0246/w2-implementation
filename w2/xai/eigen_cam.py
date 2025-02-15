# modified from https://github.com/jacobgil/pytorch-grad-cam

from w2.xai.base_cam import BaseCAM
from w2.xai.utils.svd import get_2d_projection


class EigenCAM(BaseCAM):
    """ Paper: https://arxiv.org/abs/2008.00299 """
    name = "Eigen-CAM"

    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform,
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)

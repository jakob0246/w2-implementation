"""

This script demonstrates the use of the w2 X-AI module for crack detection.
The script uses the CrackPy package to detect crack tips and the w2 X-AI module to explain the crack tip detection.

"""


import os

import torch
import requests

from matplotlib import pyplot as plt

from crackpy.crack_detection.model import get_model
from crackpy.crack_detection.detection import CrackDetection, CrackTipDetection
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.data_files import Nodemap

from w2 import XAI


class ParallelNetsWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ParallelNetsWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        """No Sigmoid and only take segmentation output. This was proposed by Grad-CAM authors."""
        x1 = self.model.inc(x)
        x2 = self.model.down1(x1)
        x3 = self.model.down2(x2)
        x4 = self.model.down3(x3)
        x5 = self.model.down4(x4)

        x5 = self.model.base(x5)

        x = self.model.up1(x5, x4)
        x = self.model.up2(x, x3)
        x = self.model.up3(x, x2)
        x = self.model.up4(x, x1)
        x = self.model.outc(x)
        return x


NODEMAP_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt'

# Download test nodemap
if not os.path.exists(NODEMAP_FILE):
    url = "https://raw.githubusercontent.com/dlr-wf/crackpy/main/test_data/crack_detection/Nodemaps/Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt"
    r = requests.get(url)
    open('Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt', 'wb').write(r.content)
    print("Downloaded test nodemap.")


# Setup
det = CrackDetection(
    side='right',
    detection_window_size=60,
    offset=(5, 0),
    angle_det_radius=10,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)

# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder="")
data = InputData(nodemap)
print(f"Force: {data.force} N")

# Interpolate data on arrays (256 x 256 pixels)
interp_disps, _ = det.interpolate(data)

# Preprocess input
input_tensor = det.preprocess(interp_disps)

# Load crack tip detector
model = get_model('ParallelNets').unet

# Detect crack tips
ct_det = CrackTipDetection(detection=det, tip_detector=model)
pred = ct_det.make_prediction(input_tensor)
ct_segmentation = ct_det.calculate_segmentation(pred)
seg_mask = torch.where(pred > 0.5, 1, 0)[0].numpy()
ct_pixels = ct_det.find_most_likely_tip_pos(pred)
ct_x, ct_y = ct_det.calculate_position_in_mm(ct_pixels)
print(f"Crack tip x [mm]: {ct_x}")
print(f"Crack tip y [mm]: {ct_y}")

# w2 X-AI
model = ParallelNetsWrapper(model)
target_layers = [
    model.model.down1,
    model.model.down2,
    model.model.down3,
    model.model.down4,
    model.model.base,
    model.model.up1,
    model.model.up2,
    model.model.up3,
    model.model.up4
]

####################################################################
# Run X-AI module
####################################################################
xai = XAI(
    model=model,
    methods=[
        "GradCAM",  # <1 seconds
        "GradCAM++",  # <1 seconds
        "ScoreCAM",  # ~60 seconds
        "EigenCAM"  # ~5 minutes
    ],
    mode="segmentation",
    target_layers=target_layers,
    params={
        "eigen_smooth": False  # For Semantic Segmentation, eigen_smooth does not seem to work well
    }
)
explanations = xai.explain(
    inp=input_tensor,
    class_num=0,
    mask=None  # seg_mask
)

# Write results
for method in xai.methods:
    # Plot
    fig, ax = plt.subplots(1, 1)
    plt.imshow(explanations[method][0], cmap='jet')
    plt.colorbar()
    plt.scatter(ct_pixels[1], ct_pixels[0], marker='x', c='black', label='Crack tip')
    plt.legend()
    plt.axis('off')
    plt.savefig(NODEMAP_FILE[:-4] + '_' + method + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


####################################################################
# Plot results
####################################################################
fig, ax = plt.subplots(3, 2, figsize=(8, 12))

# Plot input displacement
ax[0, 0].imshow(interp_disps[1], cmap="coolwarm")
ax[0, 0].scatter(ct_pixels[1], ct_pixels[0], marker='x', c='black', label='Crack tip')
ax[0, 0].axis('off')
ax[0, 0].set_title("vertical displacement")

# Plot segmentation mask
ax[0, 1].imshow(seg_mask, cmap="gray")
ax[0, 1].scatter(ct_pixels[1], ct_pixels[0], marker='x', c='black', label='Crack tip')
ax[0, 1].axis('off')
ax[0, 1].set_title("crack tip segmentation")

# Plot results of CAM methods
for cam_method, axis in zip(xai.methods, ax.flatten()[2:]):
    cam = explanations[cam_method]
    axis.imshow(cam[0, :], cmap='jet')
    axis.axis('off')
    axis.set_title(cam_method)
plt.savefig("w2_xai_crack_detection.png", dpi=300, bbox_inches="tight")
plt.close()
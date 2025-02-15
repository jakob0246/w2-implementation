import os
import cv2
from matplotlib import pyplot as plt
import w2
import warnings
import requests
from PIL import Image
import torch
import torch.nn.functional as F
import ttach as tta
import numpy as np

from crackpy.crack_detection.detection import CrackDetection, CrackTipDetection
from crackpy.crack_detection.model import get_model
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.data_files import Nodemap

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class ParallelNetsWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ParallelNetsWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        """No Sigmoid and only take segmentation output. This was proposed by Grad-CAM authors."""
        x1 = self.model.inc(x)
        x1 = F.dropout2d(x1, p=0.25, training=True)
        x2 = self.model.down1(x1)
        x2 = F.dropout2d(x2, p=0.25, training=True)
        x3 = self.model.down2(x2)
        x3 = F.dropout2d(x3, p=0.25, training=True)
        x4 = self.model.down3(x3)
        x4 = F.dropout2d(x4, p=0.25, training=True)
        x5 = self.model.down4(x4)
        x5 = F.dropout2d(x5, p=0.25, training=True)

        x5 = self.model.base(x5)
        x5 = F.dropout2d(x5, p=0.25, training=True)

        x = self.model.up1(x5, x4)
        x = F.dropout2d(x, p=0.25, training=True)
        x = self.model.up2(x, x3)
        x = F.dropout2d(x, p=0.25, training=True)
        x = self.model.up3(x, x2)
        x = F.dropout2d(x, p=0.25, training=True)
        x = self.model.up4(x, x1)
        x = F.dropout2d(x, p=0.25, training=True)
        x = self.model.outc(x)
        return x


def preprocess(tensor):
    """Dummy preprocess function that does nothing"""
    return tensor


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

mask_uint8 = 255 * np.uint8(seg_mask == 1)
mask_float = np.float32(seg_mask == 1)


####################################################################
# W2 Uncertainty estimation
####################################################################
# Wrapper with dropout for UC prediction
model = ParallelNetsWrapper(model)

# Uncertainty methods you want to evaluate
methods = [
    "dropout",
    "data-augmentation",
    "weight-noise",
    "softmax"
]

# W2 call
uc = w2.UC(
    methods=methods,
    mode="segmentation",
    model=model,
    preprocess=preprocess,
    params={
        "data-augmentation": {
            "augmentations": tta.Compose([
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1.0, 1.1])
            ]),
            "noise_factor": 0.2
        },
        "dropout": {
            "num_samples": 100
        },
        "weight-noise": {
            "num_samples": 100,
            "noise_factor": 0.05
        }
    }
)

# Estimate uncertainty
uncertainties = uc.predict([input_tensor[0]], pos_class=0, metric="std")


####################################################################
# Plot results
####################################################################
names = {
    "dropout": "TTD",
    "data-augmentation": "TTA",
    "weight-noise": "Weight Noise",
    "softmax": "Softmax"
}

fig, ax = plt.subplots(3, 2, figsize=(8, 12))

# Plot input displacement
ax[0, 0].imshow(interp_disps[1], cmap="coolwarm")
ax[0, 0].scatter(ct_pixels[1], ct_pixels[0], marker='x', c='black', label='Crack tip')
ax[0, 0].axis('off')
ax[0, 0].set_title("y-displacement")

# Plot segmentation mask
ax[0, 1].imshow(seg_mask, cmap="gray")
#ax[0, 1].scatter(ct_pixels[1], ct_pixels[0], marker='x', c='black', label='Crack tip')
ax[0, 1].axis('off')
ax[0, 1].set_title("crack tip segmentation")

# Plot results of CAM methods
for uc_method, axis in zip(methods, ax.flatten()[2:]):
    map = (np.array(uncertainties[uc_method][0]) * 255).astype(np.uint8)[:, :, np.newaxis]
    map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
    map = cv2.cvtColor(map, cv2.COLOR_RGB2BGR).astype(np.float32)
    map = Image.fromarray(map.astype("uint8"))

    # plot to image with attention map overlay
    axis.imshow(map)
    axis.axis('off')
    axis.set_title(names[uc_method])
plt.savefig(f"w2_uc_crack_detection.png", dpi=300, bbox_inches="tight")
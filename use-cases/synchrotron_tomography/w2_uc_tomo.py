import cv2
from matplotlib import pyplot as plt

import w2
import warnings
from PIL import Image
import torch
import ttach as tta
import numpy as np
from unet import UNetWithCracks
import tifffile as tiff

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def preprocess(tensor):
    """Dummy preprocess function that does nothing"""
    return tensor


# Load test data
input_data = tiff.imread("../../data/tomo/12431_RTT_320N_8bit_946x781x101_ringCor_crop_BH_resliceTop_norm0.4_crop_RAW.tif")
input_tensor = torch.from_numpy(input_data).unsqueeze(1)
target_data = tiff.imread("../../data/tomo/12431_RTT_320N_8bit_946x781x101_ringCor_crop_BH_resliceTop_norm0.4_crop_BINARY.tif")
target_tensor = torch.from_numpy(target_data)

input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

# Normalize input
std = input_tensor.std()
mean = input_tensor.mean()
input_tensor = (input_tensor - mean) / std

# random crop of 256 x 256
start = 0
input_tensor = input_tensor[50, :, start:start+256, start:start+256].unsqueeze(0)
target_tensor = target_tensor[50, start:start+256, start:start+256].unsqueeze(0)

# Load model
MODELPATH = "../../models/tomo/model_e137_l0.6860.pt"
model = UNetWithCracks()
model.load_state_dict(torch.load(MODELPATH))
model = model.eval()

## Predict
output = model(input_tensor)
normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()

sem_classes = [
    'Class 0',  # background
    'Class 1',  # cracks
    'Class 2',  # inter-metallic phase
    'Class 3'   # silicon phase
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

category = sem_class_to_idx["Class 3"]
mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
mask_uint8 = 255 * np.uint8(mask == category)
mask_float = np.float32(mask == category)


####################################################################
# W2 Uncertainty estimation
####################################################################

# Uncertainty methods you want to evaluate
methods = [
    "dropout",
    "data-augmentation",
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
            "noise_factor": 0.1
        },
        "dropout": {
            "augmentations": tta.Compose([
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 180]),
                tta.Multiply(factors=[0.95, 1.0, 1.05])
            ]),
            "num_samples": 20
        },
        "weight-noise": {
            "num_samples": 100,
            "noise_factor": 0.005
        }
    }
)

# Estimate uncertainty
uncertainties = uc.predict([input_tensor[0]], pos_class=category, metric="std")


####################################################################
# Plot results
####################################################################
names = {
    "dropout": "TTD",
    "data-augmentation": "TTA",
    "softmax": "Softmax"
}
fig, ax = plt.subplots(2, 3, figsize=(9, 6))

# Plot original image
denormalized_input = input_tensor * std + mean
ax[0, 0].imshow(Image.fromarray(denormalized_input[0, 0, :, :].detach().cpu().numpy()))
ax[0, 0].axis('off')
ax[0, 0].set_title("2D X-ray Tomography")

# Plot segmentation mask
ax[0, 1].imshow(Image.fromarray(mask_uint8), cmap='gray')
ax[0, 1].axis('off')
ax[0, 1].set_title("Segmentation")

# Plot target mask
ax[0, 2].imshow(Image.fromarray((target_tensor[0, :, :].numpy() == category)), cmap='gray')
ax[0, 2].axis('off')
ax[0, 2].set_title("Ground Truth")

image = np.concatenate([input_tensor[0, :, :, :].detach().cpu().numpy(),
                       input_tensor[0, :, :, :].detach().cpu().numpy(),
                       input_tensor[0, :, :, :].detach().cpu().numpy()]).transpose(1, 2, 0)
# Plot results of CAM methods
for uc_method, axis in zip(methods, ax.flatten()[3:]):
    map = (np.array(uncertainties[uc_method][0]) * 255).astype(np.uint8)[:, :, np.newaxis]
    map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
    map = cv2.cvtColor(map, cv2.COLOR_RGB2BGR).astype(np.float32)
    map = Image.fromarray(map.astype("uint8"))

    # plot to image with attention map overlay
    axis.imshow(map)
    axis.axis('off')
    axis.set_title(names[uc_method])
plt.savefig(f"w2_uc_tomo_class_{category}.png", dpi=600, bbox_inches="tight")
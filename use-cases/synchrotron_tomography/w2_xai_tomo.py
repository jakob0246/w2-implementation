import warnings

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch

from w2 import XAI
from w2.xai.utils.image import show_cam_on_image
from unet import UNetWithCracks
import tifffile as tiff

# Color map for matplotlib
plt.rcParams['image.cmap'] = 'coolwarm'

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

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

# Predict
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

"""target_layers = [
    model.conv1,
    model.conv11,
    model.conv12,
    model.conv2,
    model.conv21,
    model.conv3,
    model.conv31,
    model.conv32,
    model.conv4,
    model.conv41,
    model.conv42,
    model.conv5,
    model.conv51,
    model.conv52,
    model.conv53,
    model.conv54,
    model.Up2,
    model.Up3,
    model.Up4,
    model.Up5,
    model.Up51,
    model.Up52,
    model.Up53
]"""
target_layers = [
    model.Up53
]

####################################################################
# Run X-AI module
####################################################################
xai = XAI(
    model=model,
    methods=[
        "GradCAM",
        "GradCAM++",
        "ScoreCAM",
        # "EigenCAM"  # EigenCAM is not class-discriminative
    ],
    mode="segmentation",
    target_layers=target_layers,
    params={
        "use_cuda": True
    }
)
explanations = xai.explain(
    inp=input_tensor,
    class_num=category,
    mask=None  # or mask_float
)

####################################################################
# Plot results
####################################################################
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
for cam_method, axis in zip(xai.methods, ax.flatten()[3:]):
    cam = explanations[cam_method]
    cam_image = show_cam_on_image(image / 255, cam[0, :], image_weight=0)

    # plot to image with attention map overlay
    axis.imshow(Image.fromarray(cam_image))
    axis.axis('off')
    axis.set_title(cam_method)
plt.savefig(f"w2_xai_tomo_class_{category}_Up5.png", dpi=600, bbox_inches="tight")

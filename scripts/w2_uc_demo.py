import cv2
from matplotlib import pyplot as plt

import w2
import warnings
import requests
import numpy as np
from PIL import Image
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

from w2.xai.utils.image import preprocess_image, show_cam_on_image
from torchvision.transforms import Compose, Normalize, ToTensor

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def preprocess_image(
        img: np.ndarray,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy())


def preprocess(image):
    image = np.array(image)
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    return input_tensor


# Download test image
image_url = "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
image = np.array(Image.open(requests.get(image_url, stream=True).raw))
rgb_img = np.float32(image) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Taken from the torchvision tutorial
# https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
model = deeplabv3_resnet50(pretrained=True, progress=False)
model = model.eval()

model = SegmentationModelOutputWrapper(model)
output = model(input_tensor.unsqueeze(0))

normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
sem_classes = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

car_category = sem_class_to_idx["car"]
car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
car_mask_float = np.float32(car_mask == car_category)

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
)

# Estimate uncertainty
uncertainties = uc.predict([torch.tensor(image, dtype=torch.float32)], metric="counter_max")

# Write results
for method in methods:
    uncertainty_map = (np.array(uncertainties[method][0]) * 255).astype(np.uint8)[:, :, np.newaxis]
    uncertainty_map = cv2.applyColorMap(uncertainty_map, cv2.COLORMAP_JET)
    uncertainty_map = cv2.cvtColor(uncertainty_map, cv2.COLOR_RGB2BGR).astype(np.float32)
    uncertainty_map = Image.fromarray(uncertainty_map.astype("uint8"))
    uncertainty_map.save(f"w2_uc_demo_{method}.png")

####################################################################
# Plot results
####################################################################
fig, ax = plt.subplots(3, 2, figsize=(8, 8))

# Plot original image
ax[0, 0].imshow(Image.fromarray(image))
ax[0, 0].axis('off')
ax[0, 0].set_title("Original Image")

# Plot segmentation mask
ax[0, 1].imshow(Image.fromarray(car_mask_uint8), cmap='gray')
ax[0, 1].axis('off')
ax[0, 1].set_title("Segmentation Mask")

# Plot results of CAM methods
for uc_method, axis in zip(methods, ax.flatten()[2:]):
    map = uncertainties[uc_method]
    map_image = show_cam_on_image(rgb_img, map[0, :], use_rgb=True)

    # plot to image with attention map overlay
    axis.imshow(Image.fromarray(map_image))
    axis.axis('off')
    axis.set_title(uc_method)
plt.savefig("w2_uc_demo_overview.png", dpi=300, bbox_inches="tight")
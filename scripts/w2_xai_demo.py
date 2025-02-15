import warnings

import cv2
import requests

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

from w2 import XAI
from w2.xai.utils.image import show_cam_on_image, preprocess_image

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


# Download test image
image_url = "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
image = np.array(Image.open(requests.get(image_url, stream=True).raw))
rgb_img = np.float32(image) / 255
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
# Taken from the torchvision tutorial
# https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
model = deeplabv3_resnet50(pretrained=True, progress=False)
model = model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

model = SegmentationModelOutputWrapper(model)
output = model(input_tensor)

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

target_layers = [model.model.backbone.layer4]

####################################################################
# Run X-AI module
####################################################################
xai = XAI(
    model=model,
    methods=[
        "GradCAM",
        "GradCAM++",
        "ScoreCAM",
        "EigenCAM"
    ],
    mode="segmentation",
    target_layers=target_layers,
    params={
        "eigen_smooth": False  # For Semantic Segmentation, eigen_smooth does not seem to work well
    }
)
explanations = xai.explain(
    inp=input_tensor,
    class_num=car_category,
    mask=car_mask_float
)

# Write results
for method in xai.methods:
    xai_map = (np.array(explanations[method][0]) * 255).astype(np.uint8)[:, :, np.newaxis]
    xai_map = cv2.applyColorMap(xai_map, cv2.COLORMAP_JET)
    xai_map = cv2.cvtColor(xai_map, cv2.COLOR_RGB2BGR).astype(np.float32)
    xai_map = Image.fromarray(xai_map.astype("uint8"))
    xai_map.save(f"w2_xai_demo_{method}.png")

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
for cam_method, axis in zip(xai.methods, ax.flatten()[2:]):
    cam = explanations[cam_method]
    cam_image = show_cam_on_image(rgb_img, cam[0, :], use_rgb=True)

    # plot to image with attention map overlay
    axis.imshow(Image.fromarray(cam_image))
    axis.axis('off')
    axis.set_title(cam_method)
plt.savefig("w2_xai_demo_overview.png", dpi=300, bbox_inches="tight")

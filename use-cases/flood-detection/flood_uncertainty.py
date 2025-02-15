import cv2
import torch
import rasterio
import numpy as np
from PIL import Image
import kornia.augmentation as K
from torchvision import transforms
import torchvision.transforms.functional as F
import ttach as tta

import w2

from w2.uncertainty.utils.unet import UNet


# Preprocessing function
# Just use any preprocessing function that you use for evaluation, e.g, just normalization
def preprocess_eval_image(image):
    image = K.Normalize([0.6851, 0.5235], [0.0820, 0.1102])(image).squeeze()

    im_c1 = Image.fromarray(image[0].numpy()).resize((512, 512))
    im_c2 = Image.fromarray(image[1].numpy()).resize((512, 512))

    im_c1s = [F.crop(im_c1, 0, 0, 256, 256), F.crop(im_c1, 0, 256, 256, 256),
              F.crop(im_c1, 256, 0, 256, 256), F.crop(im_c1, 256, 256, 256, 256)]

    im_c2s = [F.crop(im_c2, 0, 0, 256, 256), F.crop(im_c2, 0, 256, 256, 256),
              F.crop(im_c2, 256, 0, 256, 256), F.crop(im_c2, 256, 256, 256, 256)]

    images = [torch.stack((transforms.ToTensor()(x).squeeze(),
                           transforms.ToTensor()(y).squeeze()))
              for (x, y) in zip(im_c1s, im_c2s)]
    images = torch.stack(images)

    return images


# (Optional!) Postprocessing function, just needed for the Sen1Floods11 dataset
def stitch_2_by_2_patches(arrays):
    arrays = arrays.permute(0, 2, 3, 1)
    reduced_arrays = []
    for i in range(len(arrays) // 4):
        row_1 = torch.cat((arrays[4 * i], arrays[4 * i + 1]), dim=1)
        row_2 = torch.cat((arrays[4 * i + 2], arrays[4 * i + 3]), dim=1)
        reduced_arrays.append(torch.cat((row_1, row_2)))
    reduced_arrays = torch.stack(reduced_arrays)
    reduced_arrays = reduced_arrays.permute(0, 3, 1, 2)
    return reduced_arrays


# Get and preprocess SAR image
sar_image = rasterio.open("sar_image_paraguay_1029191.tif").read()
sar_image = np.clip(sar_image, -50, 1)
sar_image = torch.tensor((sar_image + 50) / 51)

# Load model
model = UNet(n_classes=2, use_group_norm=True).cuda()
model.load_state_dict(torch.load("reference_model.pt", map_location=torch.device("cuda")))

# Uncertainty methods you want to evaluate
methods = [
    "student-ensemble",
    "dropout",
    "data-augmentation",
    "weight-noise",
    "softmax"
]

# W2 call
uc = w2.UC(
    methods,                             # Methods
    "segmentation",                      # Mode
    model,                               # Model
    preprocess_eval_image,               # Preprocessing function
    # Method Parameters
    {
        "student-ensemble": {
            "num_students": 3,
            "num_epochs": 100,
            "ensemble_train_params": {"use_dropout": False, "use_group_norm": True},
            "student_model_saves_directory": "./student_ensemble/"
        },
        "data-augmentation": {
            "noise_factor": 0.2,
            "augmentations": tta.Compose([
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 180]),
                tta.Multiply(factors=[0.95, 1.0, 1.05])
])

        },
        "dropout": {
            "num_samples": 20
        },
        "weight-noise": {
            "num_samples": 100,
            "noise_factor": 0.005
        }
    },
    postprocess=stitch_2_by_2_patches,                             # (Optional!) Postprocessing func. for model output
    evaluation_collate=lambda x: (torch.cat([a for a in x], 0))    # (Optional!) Just needed for Sen1Floods11 dataset
)

# Estimate uncertainty
uncertainties = uc.predict([sar_image], metric="counter_max")

# Write results
for i, method in enumerate(methods):
    uncertainty_map = (np.array(uncertainties[method][0]) * 255).astype(np.uint8)[:, :, np.newaxis]
    uncertainty_map = cv2.applyColorMap(uncertainty_map, cv2.COLORMAP_JET)
    uncertainty_map = cv2.cvtColor(uncertainty_map, cv2.COLOR_RGB2BGR).astype(np.float32)
    uncertainty_map = Image.fromarray(uncertainty_map.astype("uint8"))
    uncertainty_map.save(f"uncertainty_{method}.jpg")

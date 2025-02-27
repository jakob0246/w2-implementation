import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from utils import rgb2int


def _load_label_classes(label):
    """
    // Source: Lanoge //
    Combine `rgb2int` with processing class outputs.
    """

    classes = [(255, 0, 0),      # 0 - Clutter/background
               (0, 255, 0),      # 1 - Tree
               (0, 0, 255),      # 2 - Building
               (255, 255, 0),    # 3 - Car
               (0, 255, 255),    # 4 - Low vegetation
               (255, 255, 255)]  # 5 - Impervious surfaces

    if np.ndim(classes) != 0:
        rgbclasses = np.array(classes)
        classes = np.arange(len(classes))
        label = rgb2int(label, rgbclasses)
    else:
        rgbclasses = np.unique(label.reshape(-1, label.shape[2]), axis=0)
        classes = np.arange(len(rgbclasses))
        label = rgb2int(label, rgbclasses)
    return label, classes, rgbclasses


def _get_images_and_labels(preprocessed_path, split_type):
    image_paths = [path for path in Path(os.path.join(preprocessed_path, "images", split_type)).rglob('*.tif')]
    label_paths = [path for path in Path(os.path.join(preprocessed_path, "labels", split_type)).rglob('*.tif')]

    image_paths = sorted(image_paths, key=lambda x: x.stem)
    label_paths = sorted(label_paths, key=lambda x: x.stem)

    # Train images and labels
    images, labels = [], []
    for i in range(len(image_paths)):
        image = np.array(Image.open(image_paths[i]))
        label = np.array(Image.open(label_paths[i]))

        label, _, _ = _load_label_classes(label)

        images.append(torch.tensor(image, dtype=torch.float))
        labels.append(torch.tensor(label, dtype=torch.float))

        # TODO: remove
        # For testing: only load first 100 images
        # if i == 99:
        #     return images, labels

    return images, labels


def load_data(dataset_path):
    preprocessed_path = os.path.join(dataset_path, "Preprocessed")
    assert os.path.isdir(preprocessed_path), "Preprocess Potsdam dataset before using"

    train_images, train_labels = _get_images_and_labels(preprocessed_path, "train")
    val_images, val_labels = _get_images_and_labels(preprocessed_path, "val")
    test_images, test_labels = _get_images_and_labels(preprocessed_path, "test")

    train_data = list(zip(train_images, train_labels))
    val_data = list(zip(val_images, val_labels))
    test_data = list(zip(test_images, test_labels))

    return train_data, val_data, test_data

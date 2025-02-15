import torch
import argparse
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=24, type=int)

    parser.add_argument("--dataset-path", default="/ste/rnd/Flood_Datasets/sen1floods11/v1.1/", type=str)
    parser.add_argument("--gpu", default=2, type=int)
    parser.add_argument("--num-workers", default=1, type=int)

    parser.add_argument("--students", default=50, type=int, help="Size of the student models ensemble")
    parser.add_argument("--student-epochs", default=100, type=int)
    parser.add_argument("--pretrained-reference", action="store_true")

    parser.set_defaults(argument=True)
    arguments = parser.parse_args()
    return arguments


def stitch_2_by_2_patches(arrays):
    # Swap channel dimension if needed
    if len(arrays.shape) == 4 and arrays.shape[2] == arrays.shape[3]:
        arrays = arrays.permute(0, 2, 3, 1)
    reduced_arrays = []
    for i in range(len(arrays) // 4):
        row_1 = torch.cat((arrays[4 * i], arrays[4 * i + 1]), dim=1)
        row_2 = torch.cat((arrays[4 * i + 2], arrays[4 * i + 3]), dim=1)
        reduced_arrays.append(torch.cat((row_1, row_2)))
    reduced_arrays = torch.stack(reduced_arrays)
    # Swap channel dimension again if needed, to reset it
    if len(reduced_arrays.shape) == 4 and reduced_arrays.shape[1] == reduced_arrays.shape[2]:
        reduced_arrays = reduced_arrays.permute(0, 3, 1, 2)
    return reduced_arrays


def patchify_2_by_2(arrays):
    result = []
    for array in arrays:
        array = Image.fromarray(array)
        patches = [F.crop(array, 0, 0, 256, 256), F.crop(array, 0, 256, 256, 256),
                   F.crop(array, 256, 0, 256, 256), F.crop(array, 256, 256, 256, 256)]
        result.extend(patches)
    return np.array(result)

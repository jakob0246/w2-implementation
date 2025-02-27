import csv
import os

import numpy as np
import rasterio
import torch
from tqdm import tqdm


def get_flood_array(fname):
    return rasterio.open(fname).read()


def download_flood_water_data_from_list(l, desc_label):
    flood_data = []
    for (s1_image_path, label_path) in tqdm(l, desc=desc_label, unit="files", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        if not os.path.exists(s1_image_path):
            continue
        arr_x_s1 = np.nan_to_num(get_flood_array(s1_image_path))
        arr_x_s1 = np.clip(arr_x_s1, -50, 1)
        arr_x_s1 = (arr_x_s1 + 50) / 51

        arr_y = get_flood_array(label_path)
        arr_y[arr_y == -1] = 255

        # Don't add images with labels in the val dataset where all label values are invalid (-1)
        # This results in some losses and accuracies being NaN
        if np.all(arr_y == 255):
            continue

        flood_data.append((torch.tensor(arr_x_s1), torch.tensor(arr_y).squeeze()))

    return flood_data


def load_flood_data(root_path, split_type_filename, desc_label):
    split_path = os.path.join(root_path, "splits/flood_handlabeled", split_type_filename)
    s1_image_path = os.path.join(root_path, "data/flood_events/HandLabeled/S1Hand/")
    label_path = os.path.join(root_path, "data/flood_events/HandLabeled/LabelHand/")
    files = []
    with open(split_path) as f:
        for line in csv.reader(f):
            files.append(tuple((s1_image_path + line[0], label_path + line[1])))

    image_files_without_path = [file[0].split("/")[len(file[0].split("/")) - 1] for file in files]
    return download_flood_water_data_from_list(files, desc_label), image_files_without_path

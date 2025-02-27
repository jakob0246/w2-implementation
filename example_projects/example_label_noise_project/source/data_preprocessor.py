import os
import numpy as np
from PIL import Image
from pathlib import Path
from patchify import patchify

from utils import map_to_noisy_label_name


def _get_image_and_label_paths(image_dir, label_dir):
    image_paths = [path for path in Path(image_dir).rglob('*.tif')]
    label_paths = [path for path in Path(label_dir).rglob('*.tif')]

    image_paths = sorted(image_paths, key=lambda x: x.stem)
    label_paths = sorted(label_paths, key=lambda x: x.stem)

    assert len(image_paths) == 38
    assert len(label_paths) == 38

    return image_paths, label_paths


def _correct_labels(label_dir, label_paths):
    # correct invalid values in two label tiles
    # 4_12: blending of many values
    # 6_7: some 255 values replaced by 252
    label_4_12 = np.array(Image.open(label_dir + r'/top_potsdam_4_12_label.tif')).astype('uint8')
    label_4_12_corrected = np.where(label_4_12 >= 128, 255, 0)
    label_6_7 = np.array(Image.open(label_dir + r'/top_potsdam_6_7_label.tif')).astype('uint8')
    label_6_7_corrected = np.where(label_6_7 == (252, 255, 0), (255, 255, 0), label_6_7)
    assert (np.unique(label_4_12_corrected) == [0, 255]).all()
    assert (np.unique(label_6_7_corrected) == [0, 255]).all()

    # save the corrected label tiles and replace the original paths
    if not os.path.isdir(label_dir + '_corrected'):
        os.makedirs(label_dir + '_corrected')
    Image.fromarray(label_4_12_corrected.astype('uint8')).save(label_dir + r'_corrected/top_potsdam_4_12_label.tif')
    Image.fromarray(label_6_7_corrected.astype('uint8')).save(label_dir + r'_corrected/top_potsdam_6_7_label.tif')
    del label_4_12;
    del label_4_12_corrected
    del label_6_7;
    del label_6_7_corrected
    for i, path in enumerate(label_paths):
        if os.path.samefile(path, label_dir + r'/top_potsdam_4_12_label.tif'):
            label_paths[i] = Path(label_dir + r'_corrected/top_potsdam_4_12_label.tif')
        if os.path.samefile(path, label_dir + r'/top_potsdam_6_7_label.tif'):
            label_paths[i] = Path(label_dir + r'_corrected/top_potsdam_6_7_label.tif')


def _determine_dataset_split(dataset_size, train_fraction=0.6):
    """
    Probability based determination to which dataset split the image should go to, for a
    `train_fraction`-`(1 - train_fraction) / 2`-`(1 - train_fraction) / 2` split, so e.g. 60-20-20.
    Returns a list of split strings, e.g. `["val", "test", "test", "train", "test"]`.
    """

    # Create the same dataset splits each time
    np.random.seed(1)

    dataset_indices = np.arange(dataset_size)
    split_array = np.zeros(dataset_size).astype(str)

    # Train split
    train_split_size = int(dataset_size * train_fraction)
    train_split_indices = np.random.choice(dataset_size, train_split_size, replace=False)
    split_array[train_split_indices] = "train"
    dataset_indices = np.delete(dataset_indices, train_split_indices)

    # Val split
    val_fraction = (1 - train_fraction) / 2
    val_split_size = int(dataset_size * val_fraction)
    val_split_indices = np.random.choice(len(dataset_indices), val_split_size, replace=False)
    split_array[dataset_indices[val_split_indices]] = "val"
    dataset_indices = np.delete(dataset_indices, val_split_indices)

    # Test split
    split_array[dataset_indices] = "test"

    return list(split_array)


def _center_crop(array, result_size):
    """ Quadratic center crop to (result_size, result_size, #Channels). """
    input_size = array.shape[0]

    left_or_upper_offset = input_size // 2 - result_size // 2
    right_or_lower_offset = input_size // 2 + result_size // 2
    result = array[left_or_upper_offset:right_or_lower_offset, left_or_upper_offset:right_or_lower_offset, :]

    return result


def preprocess_dataset(dataset_path, num_patches):
    """ Utilizing Lanoge preprocessing. """

    preprocessed_path = os.path.join(dataset_path, "Preprocessed")
    if os.path.isdir(preprocessed_path):
        print("Dataset was already preprocessed, skipping step")
        return

    image_dir = dataset_path + r'/2_Ortho_RGB'
    label_dir = dataset_path + r'/5_Labels_all'
    assert os.path.isdir(image_dir)
    assert os.path.isdir(label_dir)

    image_paths, label_paths = _get_image_and_label_paths(image_dir, label_dir)
    _correct_labels(label_dir, label_paths)

    images_preprocessed_path = os.path.join(preprocessed_path, "images")
    labels_preprocessed_path = os.path.join(preprocessed_path, "labels")

    train_images_preprocessed_path = os.path.join(images_preprocessed_path, "train")
    val_images_preprocessed_path = os.path.join(images_preprocessed_path, "val")
    test_images_preprocessed_path = os.path.join(images_preprocessed_path, "test")
    train_labels_preprocessed_path = os.path.join(labels_preprocessed_path, "train")
    val_labels_preprocessed_path = os.path.join(labels_preprocessed_path, "val")
    test_labels_preprocessed_path = os.path.join(labels_preprocessed_path, "test")

    if not os.path.isdir(train_images_preprocessed_path):
        os.makedirs(train_images_preprocessed_path)
    if not os.path.isdir(val_images_preprocessed_path):
        os.makedirs(val_images_preprocessed_path)
    if not os.path.isdir(test_images_preprocessed_path):
        os.makedirs(test_images_preprocessed_path)
    if not os.path.isdir(train_labels_preprocessed_path):
        os.makedirs(train_labels_preprocessed_path)
    if not os.path.isdir(val_labels_preprocessed_path):
        os.makedirs(val_labels_preprocessed_path)
    if not os.path.isdir(test_labels_preprocessed_path):
        os.makedirs(test_labels_preprocessed_path)

    split_list = _determine_dataset_split(num_patches * num_patches * len(image_paths))

    for i in range(len(image_paths)):
        image = np.array(Image.open(image_paths[i]))
        label = np.array(Image.open(label_paths[i]))

        patch_size = 500
        num_channels = image.shape[2]

        # Create image patches
        image = _center_crop(image, patch_size * num_patches)
        image_patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)
        image_patches = np.reshape(image_patches, (num_patches * num_patches, patch_size, patch_size, num_channels))

        # Create label patches
        label = _center_crop(label, patch_size * num_patches)
        label_patches = patchify(label, (patch_size, patch_size, 3), step=patch_size)
        label_patches = np.reshape(label_patches, (num_patches * num_patches, patch_size, patch_size, num_channels))

        print(image_paths[i], label_paths[i])

        for j in range(image_patches.shape[0]):
            split_string = split_list[i * image_patches.shape[0] + j]

            image_patch_path = os.path.join(images_preprocessed_path, split_string,
                                            image_paths[i].stem + f"_patch_{j + 1}.tif")
            label_patch_path = os.path.join(labels_preprocessed_path, split_string,
                                            label_paths[i].stem + f"_patch_{j + 1}.tif")
            if not os.path.exists(image_patch_path):
                Image.fromarray(image_patches[j].astype("uint8")).save(image_patch_path)
            if not os.path.exists(label_patch_path):
                Image.fromarray(label_patches[j].astype("uint8")).save(label_patch_path)


def _get_image_and_label_paths_label_noise(image_dir, label_dir):
    image_paths = [path for path in Path(image_dir).rglob('*.tif')]
    label_paths = [path for path in Path(label_dir).rglob('*.png')]

    image_paths = sorted(image_paths, key=lambda x: x.stem)

    label_paths_with_image_mapping = [(label_path, map_to_noisy_label_name(label_path.stem))
                                      for label_path in label_paths]
    label_paths_with_image_mapping = sorted(label_paths_with_image_mapping, key=lambda x: x[1])
    label_paths = [label_path_with_image_mapping[0] for label_path_with_image_mapping in label_paths_with_image_mapping]

    assert len(image_paths) == 38
    assert len(label_paths) == 38

    return image_paths, label_paths


def preprocess_dataset_label_noise(image_dir, label_dir, num_patches):
    preprocessed_path = os.path.join(label_dir, "Preprocessed")
    if os.path.isdir(preprocessed_path):
        print("Dataset was already preprocessed, skipping step")
        return

    image_paths, label_paths = _get_image_and_label_paths_label_noise(image_dir, label_dir)

    images_preprocessed_path = os.path.join(preprocessed_path, "images")
    labels_preprocessed_path = os.path.join(preprocessed_path, "labels")

    train_images_preprocessed_path = os.path.join(images_preprocessed_path, "train")
    val_images_preprocessed_path = os.path.join(images_preprocessed_path, "val")
    test_images_preprocessed_path = os.path.join(images_preprocessed_path, "test")
    train_labels_preprocessed_path = os.path.join(labels_preprocessed_path, "train")
    val_labels_preprocessed_path = os.path.join(labels_preprocessed_path, "val")
    test_labels_preprocessed_path = os.path.join(labels_preprocessed_path, "test")

    if not os.path.isdir(train_images_preprocessed_path):
        os.makedirs(train_images_preprocessed_path)
    if not os.path.isdir(val_images_preprocessed_path):
        os.makedirs(val_images_preprocessed_path)
    if not os.path.isdir(test_images_preprocessed_path):
        os.makedirs(test_images_preprocessed_path)
    if not os.path.isdir(train_labels_preprocessed_path):
        os.makedirs(train_labels_preprocessed_path)
    if not os.path.isdir(val_labels_preprocessed_path):
        os.makedirs(val_labels_preprocessed_path)
    if not os.path.isdir(test_labels_preprocessed_path):
        os.makedirs(test_labels_preprocessed_path)

    split_list = _determine_dataset_split(num_patches * num_patches * len(image_paths))

    for i in range(len(image_paths)):
        image = np.array(Image.open(image_paths[i]))
        label = np.array(Image.open(label_paths[i]))

        patch_size = 500
        num_channels = image.shape[2]

        # Create image patches
        image = _center_crop(image, patch_size * num_patches)
        image_patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)
        image_patches = np.reshape(image_patches, (num_patches * num_patches, patch_size, patch_size, num_channels))

        # Create label patches
        label = _center_crop(label, patch_size * num_patches)
        label_patches = patchify(label, (patch_size, patch_size, 3), step=patch_size)
        label_patches = np.reshape(label_patches, (num_patches * num_patches, patch_size, patch_size, num_channels))

        print(image_paths[i], label_paths[i])

        for j in range(image_patches.shape[0]):
            split_string = split_list[i * image_patches.shape[0] + j]

            image_patch_path = os.path.join(images_preprocessed_path, split_string,
                                            image_paths[i].stem + f"_patch_{j + 1}.tif")
            label_patch_path = os.path.join(labels_preprocessed_path, split_string,
                                            image_paths[i].stem + f"_patch_{j + 1}.tif")
            if not os.path.exists(image_patch_path):
                Image.fromarray(image_patches[j].astype("uint8")).save(image_patch_path)
            if not os.path.exists(label_patch_path):
                Image.fromarray(label_patches[j].astype("uint8")).save(label_patch_path)

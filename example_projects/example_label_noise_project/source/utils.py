import argparse
import numpy as np
import torch


def get_arguments_baseline():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher-epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--show-test-results", action="store_true")

    parser.add_argument("--dataset-path", default="/ste/rnd/POTSDAM_LABEL_NOISE_DATA/Vanilla/Potsdam", type=str)
    parser.add_argument("--gpu", default=2, type=int)
    parser.add_argument("--num-workers", default=1, type=int)

    parser.add_argument("--students", default=3, type=int, help="Size of the student models ensemble")
    parser.add_argument("--student-epochs", default=100, type=int)
    parser.add_argument("--pretrained-teacher", action="store_true")

    parser.add_argument("--num-patches", default=5, type=int, help="The number of patches to divide "
                        "the tiles into during the preprocessing phase (e.g. 5x5)")

    parser.set_defaults(argument=True)
    arguments = parser.parse_args()
    return arguments


def get_arguments_label_noise():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher-epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--show-test-results", action="store_true")
    parser.add_argument("--evaluation", choices=["noisy-train-clean-eval", "clean-train-noisy-eval", ""], default="")

    parser.add_argument("--image-data-path", default="/ste/rnd/POTSDAM_LABEL_NOISE_DATA/Vanilla/Potsdam/2_Ortho_RGB",
                        type=str)
    parser.add_argument("--label-data-path", default="/ste/rnd/POTSDAM_LABEL_NOISE_DATA/Patch_Size_3", type=str)
    parser.add_argument("--results-path", default="/ste/rnd/Label_Noise_Study_Results/Run_1", type=str)
    parser.add_argument("--gpu", default=2, type=int)
    parser.add_argument("--num-workers", default=1, type=int)

    parser.add_argument("--students", default=3, type=int, help="Size of the student models ensemble")
    parser.add_argument("--student-epochs", default=100, type=int)
    parser.add_argument("--pretrained-teacher", action="store_true")

    parser.add_argument("--num-patches", default=5, type=int, help="The number of patches to divide "
                        "the tiles into during the preprocessing phase (e.g. 5x5)")

    parser.set_defaults(argument=True)
    arguments = parser.parse_args()
    return arguments


def int2rgb(label):
    """ Convert int image to RGB class image, (d1,d2) --> (d1,d2,3) """
    # from skimage.color import label2rgb
    # label_RGB = label2rgb(label, colors=classes, bg_label=-1)  # ~2-3(?) times slower than below

    classes = [(255, 0, 0),      # 0 - Clutter/background
               (0, 255, 0),      # 1 - Tree
               (0, 0, 255),      # 2 - Building
               (255, 255, 0),    # 3 - Car
               (0, 255, 255),    # 4 - Low vegetation
               (255, 255, 255)]  # 5 - Impervious surfaces

    if len(np.shape(label)) == 2:
        label_RGB = np.zeros(np.append(label.shape, 3))
        for c in range(len(classes)):
            label_RGB[label == c] = classes[c]
        return label_RGB
    else:
        assert np.shape(label)[2] == 3
        return label


def rgb2int(label, classes):
    """ Convert RGB image to int class image, (d1,d2,3) --> (d1,d2) """

    if len(np.shape(label)) == 3:
        assert np.shape(label)[2] == 3
        assert np.shape(classes)[1] == 3
        class_mask = [(label[:, :, 0] == r) & (label[:, :, 1] == g) & (label[:, :, 2] == b) for r, g, b in classes]
        label_ints = np.full((label.shape[0], label.shape[1]), -1)
        for i, m in enumerate(class_mask):
            label_ints[m] = i
        return label_ints
    else:
        return label


def transform2preprocessing_function(transform, is_eval=False):
    def transform_train_wrapper(data):
        image, label = data
        image, label = np.array(image), np.array(label)
        # Differentiate between label maps and label maps with multiple channels (shape = 3)
        if len(label.shape) == 2:
            transformed = transform(image=image, mask=label)
            label = transformed["mask"]
            image = transformed["image"]
        else:
            label = list(label)
            transformed = transform(image=image, masks=label)
            label = transformed["masks"]
            image = transformed["image"]
            label = torch.stack(label)
        return image, label

    def transform_eval_wrapper(image):
        image = np.array(image)
        transformed = transform(image=image)
        image = transformed["image"]
        return image

    return transform_eval_wrapper if is_eval else transform_train_wrapper


def map_to_noisy_label_name(image_file):
    mapping = {
        "tile26": "top_potsdam_2_10_RGB",
        "tile20": "top_potsdam_2_11_RGB",
        "tile22": "top_potsdam_2_12_RGB",
        "tile21": "top_potsdam_2_13_RGB",
        "tile17": "top_potsdam_2_14_RGB",
        "tile7":  "top_potsdam_3_10_RGB",
        "tile35": "top_potsdam_3_11_RGB",
        "tile19": "top_potsdam_3_12_RGB",
        "tile24": "top_potsdam_3_13_RGB",
        "tile36": "top_potsdam_3_14_RGB",
        "tile15": "top_potsdam_4_10_RGB",
        "tile28": "top_potsdam_4_11_RGB",
        "tile18": "top_potsdam_4_12_RGB",
        "tile27": "top_potsdam_4_13_RGB",
        "tile16": "top_potsdam_4_14_RGB",
        "tile6":  "top_potsdam_4_15_RGB",
        "tile10": "top_potsdam_5_10_RGB",
        "tile25": "top_potsdam_5_11_RGB",
        "tile1":  "top_potsdam_5_12_RGB",
        "tile3":  "top_potsdam_5_13_RGB",
        "tile29": "top_potsdam_5_14_RGB",
        "tile34": "top_potsdam_5_15_RGB",
        "tile12": "top_potsdam_6_10_RGB",
        "tile31": "top_potsdam_6_11_RGB",
        "tile30": "top_potsdam_6_12_RGB",
        "tile37": "top_potsdam_6_13_RGB",
        "tile32": "top_potsdam_6_14_RGB",
        "tile33": "top_potsdam_6_15_RGB",
        "tile2":  "top_potsdam_6_7_RGB",
        "tile9":  "top_potsdam_6_8_RGB",
        "tile13": "top_potsdam_6_9_RGB",
        "tile23": "top_potsdam_7_10_RGB",
        "tile11": "top_potsdam_7_11_RGB",
        "tile14": "top_potsdam_7_12_RGB",
        "tile5":  "top_potsdam_7_13_RGB",
        "tile0":  "top_potsdam_7_7_RGB",
        "tile4":  "top_potsdam_7_8_RGB",
        "tile8":  "top_potsdam_7_9_RGB"
    }

    return mapping[image_file]

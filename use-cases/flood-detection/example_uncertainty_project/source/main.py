import os
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["NUMEXPR_NUM_THREADS"] = str(1)
os.environ["OMP_NUM_THREADS"] = str(1)

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import w2

from data_loader import load_flood_data
from data_augmentor import (preprocess_train_data, preprocess_eval_data, preprocess_eval_image, InMemoryDataset,
                            preprocess_train_data_uc)
from unet import UNet
from trainer import train
from evaluator import evaluate
from results_generator import generate_example_uncertainty_maps
from utils import get_arguments, stitch_2_by_2_patches, patchify_2_by_2


# Prerequisites
args = get_arguments()
model_save_filename = f"sen1floods11_{args.epochs}_epochs.pt"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Load data
tqdm.write("\nLoading files ...\n")

# Get training data
train_data, _ = load_flood_data(args.dataset_path, "flood_train_data.csv", "Training images")
train_images = [train_image_and_label[0] for train_image_and_label in train_data]
train_dataset = InMemoryDataset(train_data, preprocess_train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers, collate_fn=None, pin_memory=True)

# Get validation data
val_data, val_images_filenames = load_flood_data(args.dataset_path, "flood_valid_data.csv", "Validation images")
val_images = [val_image_and_label[0] for val_image_and_label in val_data]
val_dataset = InMemoryDataset(val_data, preprocess_eval_data)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=lambda x: (torch.cat([a[0] for a in x], 0),
                                                               torch.cat([a[1] for a in x], 0)),
                                         pin_memory=True)

# Get test data
test_data, test_images_filenames = load_flood_data(args.dataset_path, "flood_test_data.csv", "Test images")
test_images = [test_image_and_label[0] for test_image_and_label in test_data]
test_dataset = InMemoryDataset(test_data, preprocess_eval_data)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=lambda x: (torch.cat([a[0] for a in x], 0),
                                                               torch.cat([a[1] for a in x], 0)),
                                         pin_memory=True)

# Train model if not pretrained_reference or load state dictionary
model = UNet().cuda()
model_saves_path = os.path.join("..", "model_saves", "reference", model_save_filename)

# Combine criterion, optimizer and scheduler parameters into one dictionary
train_params = {
    "criterion": nn.CrossEntropyLoss,
    "optimizer": torch.optim.AdamW,
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "lr": 1e-5,
    "batch_size": args.batch_size,
    "weight": [1, 2],
    "ignore_label_index": 255,
    "scheduler_T_0": len(train_loader) * 10,
    "scheduler_T_mult": 2,
    "collate_fn": lambda x: (torch.cat([a for a in x], 0))
}

train_stats, val_stats = {}, {}
if not args.pretrained_reference:
    tqdm.write(f"\nTraining reference model ...\n")
    train_stats, val_stats = train(model, train_loader, val_loader, args.epochs, train_params)
    torch.save(model.state_dict(), model_saves_path)
else:
    if os.path.exists(model_saves_path):
        tqdm.write("\nPretrained reference model found, loading model ...\n")
        model.load_state_dict(torch.load(model_saves_path, map_location=torch.device("cuda")))

# Evaluate trained model
train_eval_dataset = InMemoryDataset(train_data, preprocess_eval_data)
train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset, batch_size=4, shuffle=False,
                                                num_workers=args.num_workers,
                                                collate_fn=lambda x: (torch.cat([a[0] for a in x], 0),
                                                                      torch.cat([a[1] for a in x], 0)),
                                                pin_memory=True)

train_metrics, train_outputs = evaluate(model, train_eval_loader, None, use_seed=True)
val_metrics, val_outputs = evaluate(model, val_loader, None, use_seed=True)
test_metrics, test_outputs = evaluate(model, test_loader, None, use_seed=True)
tqdm.write(f"-> Train:      IoU {train_metrics['iou'] : .4f} // Bal. Acc. {train_metrics['accuracy'] : .4f}")
tqdm.write(f"-> Validation: IoU {val_metrics['iou'] : .4f} // Bal. Acc. {val_metrics['accuracy'] : .4f}")
tqdm.write(f"-> Test:       IoU {test_metrics['iou'] : .4f} // Bal. Acc. {test_metrics['accuracy'] : .4f}\n")

# Add ignore label index masks to train parameters for uncertainty estimation
# For most other use-cases for w2, this can probably be ignored
test_labels = np.array([np.array(image_and_label[1]) for image_and_label in test_data])
ignore_masks_input_imgs = np.zeros_like(test_labels)
ignore_masks_input_imgs[test_labels == 255] = 1
ignore_masks_input_imgs = patchify_2_by_2(ignore_masks_input_imgs)

train_labels = np.array([np.array(image_and_label[1]) for image_and_label in train_data])
ignore_masks_train_imgs = np.zeros_like(train_labels)
ignore_masks_train_imgs[train_labels == 255] = 1
ignore_masks_train_imgs = patchify_2_by_2(ignore_masks_train_imgs)

# Change training parameters for student ensemble uncertainty estimation
train_params["ignore_masks_input_imgs"] = ignore_masks_input_imgs
train_params["ignore_masks_train_imgs"] = ignore_masks_train_imgs
train_params["use_dropout"] = False
train_params["use_group_norm"] = True
train_params["weight"] = [1, 1]

# Estimate Uncertainty
tqdm.write("Estimating uncertainty ...")

uc = w2.UC(
    # Methods
    ["student-ensemble", "dropout", "data-augmentation", "weight-noise", "softmax"],
    "segmentation",                      # Mode
    model,                               # Model
    preprocess_eval_image,               # Preprocessing function
    # Method Parameters
    {
        "student-ensemble": {
            "num_students": args.students,                            # Size of the ensemble
            "num_epochs": args.student_epochs,                        # Epochs for training one student
            "ensemble_train_images": train_images,                    # Data on which the students should be trained on
            "ensemble_train_preprocess": preprocess_train_data_uc,    # Preprocessing function for training data
            "ensemble_train_params": train_params,                    # Student model training parameters
            # Optionally use `student_model_saves_directory`
        },
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
    },
    postprocess=stitch_2_by_2_patches,                                # Postprocessing function for model output
    evaluation_collate=train_params["collate_fn"]                     # Collate function during evaluation time
)

# Perform Evaluations
tqdm.write("\nPerforming Evaluations ...")

# Test indices of interesting images
test_indices = [46, 32, 53, 50, 65]

test_images = [test_images[i] for i in test_indices]
test_labels = test_labels[test_indices]

# Estimate uncertainty
uncertainties = uc.predict(test_images, metric="counter_max", labels=test_labels, label_preprocess=patchify_2_by_2,
                           ignore_index=train_params["ignore_label_index"])

# Generate calibration curves
# uc.generate_calibration_curves(2, test_data, ignore_index=train_params["ignore_label_index"])

# Generate uncertainty vs. error traces
# uc.evaluate_uncertainty_vs_error(test_data, metric="std", label_preprocess=patchify_2_by_2,
#                                  ignore_index=train_params["ignore_label_index"])

# Generate metric correlation plots
# uc.evaluate_metric_correlation(test_data, label_preprocess=patchify_2_by_2,
#                                ignore_index=train_params["ignore_label_index"])

# Generate method correlation plots
# uc.evaluate_method_correlation(test_data)

# Plot example uncertainty heatmaps

test_data = [test_data[i] for i in test_indices]
test_images_filenames = np.array(test_images_filenames)[test_indices]

test_outputs = np.argmax(test_outputs, axis=1)
test_outputs = np.array(stitch_2_by_2_patches(torch.tensor(test_outputs)))[test_indices]

generate_example_uncertainty_maps(test_data, test_outputs, uncertainties[0][:5], test_images_filenames)

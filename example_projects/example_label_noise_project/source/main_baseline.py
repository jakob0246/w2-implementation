import os
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["NUMEXPR_NUM_THREADS"] = str(1)
os.environ["OMP_NUM_THREADS"] = str(1)

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import W2

from unet import UNet
from trainer import train
from evaluator import evaluate
from data_loader import load_data
from data_preprocessor import preprocess_dataset
from utils import get_arguments_baseline, transform2preprocessing_function
from data_augmentor import PotsdamDataset, determine_mean_and_std, transform_train, transform_eval
from results_generator import generate_convergence_plots, generate_example_uncertainty_maps, generate_uncertainty_images


# Prerequisites
args = get_arguments_baseline()
model_save_filename = f"teacher_{args.teacher_epochs}_epochs.pt"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Preprocess dataset
tqdm.write("\nPreprocessing dataset ...\n")
# Create patches and save them to `Preprocessed` directory within the datasets folder
preprocess_dataset(args.dataset_path, args.num_patches)

# Load data
tqdm.write("\nLoading files ...\n")

train_data, val_data, test_data = load_data(args.dataset_path)
tqdm.write(f"Loaded dataset with a split: train {len(train_data)} // val {len(val_data)} // test {len(test_data)} "
           f"(total {len(train_data) + len(val_data) + len(test_data)})\n")

train_dataset = PotsdamDataset(train_data, transform_train)
val_dataset = PotsdamDataset(val_data, transform_eval)
test_dataset = PotsdamDataset(test_data, transform_eval)

# For correct determination mean has to be set to 0, and std to 1 within transforms
train_mean, train_std = determine_mean_and_std(train_dataset)
val_mean, val_std = determine_mean_and_std(val_dataset)
# print("train_mean", train_mean, "//", "train_std", train_std)
# train_mean [0.33875653 0.36236152 0.33635044] // train_std [0.13994731 0.13818924 0.14388752]
# print("val_mean", val_mean, "//", "val_std", val_std)
# val_mean [0.33896482 0.362971   0.33739454] // val_std [0.14057183 0.13951756 0.1442995]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=args.num_workers,
                                         pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=args.num_workers,
                                          pin_memory=True)

# Train model if not pretrained or load state dictionary

# Combine criterion, optimizer and scheduler parameters into one dictionary
train_params = {
    "criterion": nn.CrossEntropyLoss,
    "optimizer": torch.optim.SGD,
    "lr": 0.01,
    "weight": [4, 3, 2, 4, 1, 1],
    "use_dropout": False
}

model = UNet(use_dropout=train_params["use_dropout"]).cuda()

model_saves_path = os.path.join("..", "model_saves", "teacher", model_save_filename)
# TODO: Make nicer
assert(os.path.exists(os.path.join("..", "model_saves", "teacher")))

train_stats, val_stats = {}, {}
if not args.pretrained_teacher:
    tqdm.write(f"Training teacher model ...\n")
    train_stats, val_stats = train(model, train_loader, val_loader, args.teacher_epochs, train_params)
    torch.save(model.state_dict(), model_saves_path)
else:
    if os.path.exists(model_saves_path):
        tqdm.write("Pretrained teacher model found, loading model ...\n")
        model.load_state_dict(torch.load(model_saves_path, map_location=torch.device("cuda")))
    else:
        raise RuntimeError("Even though `pretrained-teacher` equals `True`, no pretrained model was found!")

# Evaluate trained model
train_eval_dataset = PotsdamDataset(train_data, transform_eval)
train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset, batch_size=4, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)

train_metrics, train_outputs = evaluate(model, train_eval_loader, None, use_seed=False)
val_metrics, val_outputs = evaluate(model, val_loader, None, use_seed=False)
test_metrics, test_outputs = evaluate(model, test_loader, None, use_seed=False)
tqdm.write(f"-> Train:      mIoU {train_metrics['miou'] : .4f} // Bal. Acc. {train_metrics['accuracy'] : .4f}")
tqdm.write(f"-> Validation: mIoU {val_metrics['miou'] : .4f} // Bal. Acc. {val_metrics['accuracy'] : .4f}")
# Only show test results in the end
if args.show_test_results:
    tqdm.write(f"-> Test:       mIoU {test_metrics['miou'] : .4f} // Bal. Acc. {test_metrics['accuracy'] : .4f}")

# Estimate Uncertainty
tqdm.write("\nEstimating uncertainty ...")

train_images = [image_and_label[0] for image_and_label in train_data]
val_images = [image_and_label[0] for image_and_label in val_data]
test_images = [image_and_label[0] for image_and_label in test_data]
test_labels = [image_and_label[1] for image_and_label in test_data]

uc = W2.UC(
    ["student-ensemble"],                # Methods
    "segmentation",                      # Mode
    model,                               # Model
    transform2preprocessing_function(    # Preprocessing function
        transform_eval, is_eval=True
    ),
    # Method Parameters
    {
        "student-ensemble": {
            "num_students": args.students,                            # Size of the ensemble
            "num_epochs": args.student_epochs,                        # Epochs for training one student
            "num_classes": 6,                                         # Number of classes
            "ensemble_train_images": train_images,                    # Data on which the students should be trained on
            "ensemble_train_preprocess":
                transform2preprocessing_function(transform_train),    # Preprocessing function for training data
            "ensemble_train_params": train_params                     # Student model training parameters
            # Optionally use `student_model_saves_directory`
        }
    }
)

test_indices = [2, 3, 58, 66, 97, 126]

test_data = [test_data[i] for i in test_indices]
test_images = [test_images[i] for i in test_indices]
test_labels = [test_labels[i] for i in test_indices]
test_outputs = test_outputs[test_indices]

uncertainties = uc.predict(test_images)

# Generate calibration curves for the validation data split
# uc.generate_calibration_curves("student-ensemble", val_data)

# Results generation
tqdm.write("\nPlotting results ...\n")

# If model was trained during this run
if not args.pretrained_teacher:
    generate_convergence_plots(train_stats, val_stats)

# generate_example_uncertainty_maps(test_data[:5], test_outputs[:5], uncertainties[0, :5], is_val=True)

generate_uncertainty_images(test_data, test_outputs, uncertainties[0][0:len(test_indices)])

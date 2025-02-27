import os
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["NUMEXPR_NUM_THREADS"] = str(1)
os.environ["OMP_NUM_THREADS"] = str(1)

import torch
import torch.nn as nn
from tqdm import tqdm
import itertools
from datetime import datetime

import W2

from unet import UNet
from trainer import train
from evaluator import evaluate
from data_loader import load_data
from data_preprocessor import preprocess_dataset_label_noise
from utils import get_arguments_label_noise, transform2preprocessing_function
from data_augmentor import PotsdamDataset, transform_train, transform_eval
from results_generator import (generate_convergence_plots, generate_example_uncertainty_maps, write_teacher_statistics,
                               generate_uncertainty_images_noisy)


# Prerequisites
args = get_arguments_label_noise()
model_save_filename = f"teacher_{args.teacher_epochs}_epochs.pt"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Label noise parameters
seeds = [0, 1, 2]
pixeltypes = ['singlepixel', 'superpixel']
noises = [0.05, 0.25, 0.45]
selection_methods = ['random', 'high_uncertainty', 'near_boundaries']
relabel_methods = ['random', 'confusion_matrix', 'local_neighbors', 'global_neighbors', 'similar_appearance']

# Generate all label noise combinations
noise_combinations = list(itertools.product(seeds, pixeltypes, noises, selection_methods, relabel_methods))
num_combinations = len(noise_combinations)

# Loop over all combinations
time_per_combination = None
for i, noise_combination in enumerate(noise_combinations):
    seed, pixeltype, noise, selection_method, relabel_method = noise_combination
    print(f"\nGenerating combination {i + 1}/{num_combinations}:")
    print(f"\tseed {seed} // pixeltype \"{pixeltype}\" // noise {noise} // selection-method \"{selection_method}\" "
          f"// relabel-method \"{relabel_method}\"")

    if time_per_combination is not None:
        print("\nEstimated time left:", str(time_per_combination * (num_combinations - i)))

    timestamp_begin = datetime.now()

    superpixel_directory_string = "superpixel_n6000_c10-0" if pixeltype == "superpixel" else "singlepixel"
    combination_path_suffix = os.path.join(f"seed-{seed}", superpixel_directory_string, f"noise-{noise}",
                                           f"selection-{selection_method}", f"relabel-{relabel_method}")
    combination_label_data_path = os.path.join(args.label_data_path, "labelnoise", combination_path_suffix)

    combination_results_save_path = os.path.join(args.results_path, combination_path_suffix)

    # if os.path.isdir(combination_results_save_path):
    #     print("\nCombination already processed, continuing ...")
    #     continue

    # Preprocess dataset
    tqdm.write("\nPreprocessing dataset ...\n")
    # Create patches and save them to `Preprocessed` directory within the label noise data folder
    preprocess_dataset_label_noise(args.image_data_path, combination_label_data_path, args.num_patches)

    # Load data
    tqdm.write("\nLoading files ...\n")

    if args.evaluation == "noisy-train-clean-eval":
        train_data, val_data, test_data = load_data("/ste/rnd/POTSDAM_LABEL_NOISE_DATA/Vanilla/Potsdam")
    else:
        train_data, val_data, test_data = load_data(combination_label_data_path)

    tqdm.write(f"Loaded dataset with a split: train {len(train_data)} // val {len(val_data)} // test {len(test_data)} "
               f"(total {len(train_data) + len(val_data) + len(test_data)})\n")

    train_dataset = PotsdamDataset(train_data, transform_train)
    val_dataset = PotsdamDataset(val_data, transform_eval)
    test_dataset = PotsdamDataset(test_data, transform_eval)

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
        "optimizer": torch.optim.AdamW,
        "lr": 0.0001,
        "weight": [4, 3, 2, 4, 1, 1],
        "use_dropout": False
    }

    model = UNet(use_dropout=train_params["use_dropout"]).cuda()

    model_saves_dir_path = os.path.join(combination_results_save_path, "model_saves", "teacher")
    if not os.path.isdir(model_saves_dir_path):
        os.makedirs(model_saves_dir_path)
    model_saves_file_path = os.path.join(model_saves_dir_path, model_save_filename)

    train_stats, val_stats = {}, {}
    if not args.pretrained_teacher:
        tqdm.write(f"Training teacher model ...\n")
        train_stats, val_stats = train(model, train_loader, val_loader, args.teacher_epochs, train_params)
        torch.save(model.state_dict(), model_saves_file_path)
    else:
        if os.path.exists(model_saves_file_path):
            tqdm.write("Pretrained teacher model found, loading model ...\n")
            if args.evaluation == "clean-train-noisy-eval":
                model.load_state_dict(torch.load("/ste/rnd/Label_Noise_Study_Results/Tuning/"
                                                 "val_miou_0_6191_500x500_regression_5_patches/teacher_100_epochs.pt",
                                                 map_location=torch.device("cuda")))
            else:
                model.load_state_dict(torch.load(model_saves_file_path, map_location=torch.device("cuda")))
        else:
            raise RuntimeError("Even though `pretrained-teacher` equals `True`, no pretrained model was found!")

    # Evaluate trained model
    train_eval_dataset = PotsdamDataset(train_data, transform_eval)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset, batch_size=4, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)

    train_metrics, train_outputs = evaluate(model, train_eval_loader, None, use_seed=True)
    val_metrics, val_outputs = evaluate(model, val_loader, None, use_seed=True)
    test_metrics, test_outputs = evaluate(model, test_loader, None, use_seed=True)
    tqdm.write(f"-> Train:      mIoU {train_metrics['miou'] : .4f} // Bal. Acc. {train_metrics['accuracy'] : .4f}")
    tqdm.write(f"-> Validation: mIoU {val_metrics['miou'] : .4f} // Bal. Acc. {val_metrics['accuracy'] : .4f}")
    # Only show test results in the end
    if args.show_test_results:
        tqdm.write(f"-> Test:       mIoU {test_metrics['miou'] : .4f} // Bal. Acc. {test_metrics['accuracy'] : .4f}")

    # Write evaluation results in csv
    if args.evaluation == "noisy-train-clean-eval":
        write_teacher_statistics(noise_combination, train_metrics, val_metrics, test_metrics,
                                 "/home/ludw_ja/Results/Flood Prototype/Label Noise/Train_Noisy_Data_Eval_Clean_Data")
        continue
    elif args.evaluation == "clean-train-noisy-eval":
        write_teacher_statistics(noise_combination, train_metrics, val_metrics, test_metrics,
                                 "/home/ludw_ja/Results/Flood Prototype/Label Noise/Train_Clean_Data_Eval_Noisy_Data")
        continue

    # Estimate Uncertainty
    tqdm.write("\nEstimating uncertainty ...")

    train_images = [image_and_label[0] for image_and_label in train_data]
    val_images = [image_and_label[0] for image_and_label in val_data]
    test_images = [image_and_label[0] for image_and_label in test_data]
    test_labels = [image_and_label[1] for image_and_label in test_data]

    student_model_saves_dir = os.path.join(combination_results_save_path, "model_saves", "students")
    if not os.path.isdir(student_model_saves_dir):
        os.makedirs(student_model_saves_dir)

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
                "ensemble_train_images": train_images,                    # Data on which the students should be trained
                "ensemble_train_preprocess":
                    transform2preprocessing_function(transform_train),    # Preprocessing function for training data
                "ensemble_train_params": train_params,                    # Student model training parameters
                "student_model_saves_directory": student_model_saves_dir
            }
        }
    )

    test_indices = [2]

    test_data = [test_data[i] for i in test_indices]
    test_images = [test_images[i] for i in test_indices]
    test_labels = [test_labels[i] for i in test_indices]
    test_outputs = test_outputs[test_indices]

    uncertainties = uc.predict(test_images)

    plots_save_dir = os.path.join(combination_results_save_path, "results")
    if not os.path.isdir(plots_save_dir):
        os.makedirs(plots_save_dir)

    # Generate calibration curves for the validation data split
    # uc.generate_calibration_curves("student-ensemble", test_data, save_dir=plots_save_dir)

    # Results generation
    tqdm.write("\nGenerating results ...\n")

    # If model was trained during this run
    if not args.pretrained_teacher:
        generate_convergence_plots(train_stats, val_stats, save_dir=plots_save_dir)

    # generate_example_uncertainty_maps(test_data[:5], test_outputs[:5], uncertainties[0, :5],
    #                                   save_dir=plots_save_dir, is_val=True)

    # _, _, test_data_noisy = load_data(combination_label_data_path)
    # test_labels_noisy = [image_and_label[1] for image_and_label in test_data_noisy]
    # test_labels_noisy = [test_labels_noisy[i] for i in test_indices]
    # generate_uncertainty_images_noisy(test_images, test_labels_noisy, test_outputs,
    #                                   uncertainties[0][0:len(test_indices)], plots_save_dir)

    # write_teacher_statistics(noise_combination, train_metrics, val_metrics, test_metrics, args.results_path)

    time_per_combination = datetime.now() - timestamp_begin

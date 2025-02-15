import torch
from tqdm import tqdm

from metrics import compute_iou, compute_balanced_accuracy
from evaluator import evaluate


def train_epoch(network, train_loader, epoch, n_epochs, criterion, optimizer, scheduler):
    network = network.train()

    intermediate_loss = 0
    intermediate_iou = 0
    intermediate_bal_acc = 0
    count = 0
    with tqdm(train_loader, unit="batches", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as epoch_stats:
        for (images, labels) in epoch_stats:
            epoch_stats.set_description(f"Epoch {epoch + 1}/{n_epochs}")

            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            output = network(images)
            loss = criterion(output, labels.long().cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()

            intermediate_loss += loss.item()
            intermediate_iou += compute_iou(output, labels).cpu()
            intermediate_bal_acc += compute_balanced_accuracy(output, labels).cpu()
            count += 1

            epoch_stats.set_postfix_str(f"Loss {intermediate_loss / count : .4f} // IoU {intermediate_iou / count : .4f} // Bal. Acc. {intermediate_bal_acc / count : .4f}")

    metrics = {"loss": intermediate_loss / count, "iou": intermediate_iou / count, "accuracy": intermediate_bal_acc / count}
    return metrics


def _prepare_training(model, train_params):
    criterion = train_params["criterion"](
        weight=torch.tensor(train_params["weight"]).float().cuda(),
        ignore_index=train_params["ignore_label_index"]
    )
    optimizer = train_params["optimizer"](model.parameters(), lr=train_params["lr"])
    scheduler = train_params["scheduler"](
        optimizer,
        train_params["scheduler_T_0"],
        train_params["scheduler_T_mult"],
    )
    return criterion, optimizer, scheduler


def train(network, train_loader, val_loader, n_epochs, train_params):
    criterion, optimizer, scheduler = _prepare_training(network, train_params)

    train_statistics = {
        "losses": [],
        "ious": [],
        "accuracies": []
    }
    val_statistics = {
        "losses": [],
        "ious": [],
        "accuracies": []
    }

    for epoch in range(n_epochs):
        # Train for 1 epoch and validate on validation data
        train_metrics = train_epoch(network, train_loader, epoch, n_epochs, criterion, optimizer, scheduler)
        val_metrics, _ = evaluate(network, val_loader, criterion)

        tqdm.write(f"\t-> Validation: Loss {val_metrics['loss'] : .4f} // IoU {val_metrics['iou'] : .4f} // Bal. Acc. {val_metrics['accuracy'] : .4f}\n")

        train_statistics["losses"].append(train_metrics["loss"])
        train_statistics["ious"].append(train_metrics["iou"])
        train_statistics["accuracies"].append(train_metrics["accuracy"])
        val_statistics["losses"].append(val_metrics["loss"])
        val_statistics["ious"].append(val_metrics["iou"])
        val_statistics["accuracies"].append(val_metrics["accuracy"])

    return train_statistics, val_statistics

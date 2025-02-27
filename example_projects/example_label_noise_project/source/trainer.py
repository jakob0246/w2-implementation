from tqdm import tqdm

from metrics import compute_mean_iou, compute_balanced_accuracy_mc
from evaluator import evaluate


def train_epoch(network, train_loader, epoch, n_epochs, criterion, optimizer):
    network = network.train()

    intermediate_loss = 0
    intermediate_miou = 0
    intermediate_bal_acc = 0
    count = 0
    with tqdm(train_loader, unit="batch", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as epoch_stats:
        for (images, labels) in epoch_stats:
            epoch_stats.set_description(f"Epoch {epoch + 1}/{n_epochs}")

            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            output = network(images)
            loss = criterion(output, labels.long().cuda())
            loss.backward()
            optimizer.step()

            intermediate_loss += loss.item()
            intermediate_miou += compute_mean_iou(output, labels).cpu()
            intermediate_bal_acc += compute_balanced_accuracy_mc(output, labels)
            count += 1

            epoch_stats.set_postfix_str(f"Loss {intermediate_loss / count : .4f} // "
                                        f"mIoU {intermediate_miou / count : .4f} // "
                                        f"Bal. Acc. {intermediate_bal_acc / count : .4f}")

    metrics = {"loss": intermediate_loss / count, "miou": intermediate_miou / count,
               "accuracy": intermediate_bal_acc / count}
    return metrics


def _prepare_training(model, train_params):
    criterion = train_params["criterion"]()
    optimizer = train_params["optimizer"](model.parameters(), lr=train_params["lr"])
    return criterion, optimizer


def train(network, train_loader, val_loader, n_epochs, train_params):
    criterion, optimizer = _prepare_training(network, train_params)

    train_statistics = {
        "losses": [],
        "mious": [],
        "accuracies": []
    }
    val_statistics = {
        "losses": [],
        "mious": [],
        "accuracies": []
    }

    for epoch in range(n_epochs):
        # Train for 1 epoch and validate on validation data
        train_metrics = train_epoch(network, train_loader, epoch, n_epochs, criterion, optimizer)
        val_metrics, _ = evaluate(network, val_loader, criterion)

        tqdm.write(f"\t-> Validation: Loss {val_metrics['loss'] : .4f} // mIoU {val_metrics['miou'] : .4f} // Bal. Acc. {val_metrics['accuracy'] : .4f}\n")

        train_statistics["losses"].append(train_metrics["loss"])
        train_statistics["mious"].append(train_metrics["miou"])
        train_statistics["accuracies"].append(train_metrics["accuracy"])
        val_statistics["losses"].append(val_metrics["loss"])
        val_statistics["mious"].append(val_metrics["miou"])
        val_statistics["accuracies"].append(val_metrics["accuracy"])

    return train_statistics, val_statistics

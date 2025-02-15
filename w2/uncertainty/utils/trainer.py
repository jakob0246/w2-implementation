import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import compute_mean_iou, compute_balanced_accuracy


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        # print(squared_errors.shape, y_pred.shape, y_true.shape, self.weights.unsqueeze(1).shape)
        squared_errors = squared_errors.permute(0, 2, 3, 1)
        weighted_squared_errors = squared_errors * self.weights.cuda()
        weighted_squared_errors = weighted_squared_errors.permute(0, 3, 1, 2)
        loss = torch.mean(weighted_squared_errors)
        return loss


def train_epoch(model, train_loader, optimizer, criterion, num_classes, ignore_index, ignore_masks, scheduler=None):
    model = model.train()

    intermediate_loss = 0
    intermediate_iou = 0
    intermediate_bal_acc = 0
    count = 0
    for batch, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        output = model(images)

        if ignore_masks is not None:
            # If `ignore_masks` is true, ignore indices are coded in labels as NaNs
            ignore_masks_from_nan = torch.isnan(labels)
            output[ignore_masks_from_nan] = 0
            labels[ignore_masks_from_nan] = 0

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        intermediate_loss += loss.item()
        intermediate_iou += compute_mean_iou(output, labels, num_classes, ignore_index).cpu()
        intermediate_bal_acc += compute_balanced_accuracy(output, labels, ignore_index).cpu()
        count += 1

    metrics = {"loss": intermediate_loss / count,
               "iou": intermediate_iou / count,
               "accuracy": intermediate_bal_acc / count}

    return metrics


def _prepare_training(model, train_params, train_loader, num_classes):
    train_params_keys = list(train_params.keys())

    weight = train_params.get("weight")
    criterion = WeightedMSELoss(torch.tensor(weight if weight is not None else [1] * num_classes))

    optimizer = train_params["optimizer"](model.parameters(), lr=train_params["lr"])

    scheduler = None
    if "scheduler" in train_params_keys:
        scheduler_T_0 = train_params["scheduler_T_0"] \
            if "scheduler_T_0" in train_params_keys else len(train_loader) * 10
        scheduler_T_mult = train_params["scheduler_T_mult"] if "scheduler_T_mult" in train_params_keys else 1
        scheduler = train_params["scheduler"](
            optimizer,
            scheduler_T_0,
            T_mult=scheduler_T_mult
        )

    return criterion, optimizer, scheduler


def train(model, train_loader, n_epochs, num_classes, train_params):
    criterion, optimizer, scheduler = _prepare_training(model, train_params, train_loader, num_classes)

    with tqdm(range(n_epochs), unit="epoch", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as epoch_stats:
        for epoch in epoch_stats:
            epoch_stats.set_description(f"Epoch {epoch + 1}/{n_epochs}")
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, num_classes,
                                        train_params.get("ignore_label_index"),
                                        train_params.get("ignore_masks_train_imgs"), scheduler=scheduler)
            epoch_stats.set_postfix_str(f"Loss {train_metrics['loss'] : .4f} // "
                                        f"{'IoU' if num_classes == 2 else 'mIoU'} {train_metrics['iou'] : .4f} // "
                                        f"Bal. Acc. {train_metrics['accuracy'] : .4f}")

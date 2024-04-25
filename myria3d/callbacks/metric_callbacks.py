from pytorch_lightning import Callback, LightningModule, Trainer
import torch
from torchmetrics import Accuracy


class ModelDetailedMetrics(Callback):
    def __init__(self, num_classes=7):
        self.num_classes = num_classes

    def on_fit_start(self, trainer, pl_module) -> None:
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_acc_class = Accuracy(
            task="multiclass", num_classes=self.num_classes, average=None
        )

        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc_class = Accuracy(
            task="multiclass", num_classes=self.num_classes, average=None
        )

    def on_test_start(self, trainer, pl_module) -> None:
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc_class = Accuracy(
            task="multiclass", num_classes=self.num_classes, average=None
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        logits = outputs["logits"]
        targets = outputs["targets"]
        preds = torch.argmax(logits.detach(), dim=1)
        self.train_acc.to(preds.device)(preds, targets)
        self.train_acc_class.to(preds.device)(preds, targets)

    def on_train_epoch_end(self, trainer, pl_module):
        # global
        pl_module.log(
            "train/acc", self.train_acc, on_epoch=True, on_step=False, metric_attribute="train/acc"
        )
        # per class
        class_names = pl_module.hparams.classification_dict.values()
        accuracies = self.train_acc_class.compute()
        self.log_all_class_metrics(accuracies, class_names, "acc", "train")

    def on_validation_batch_end(self, valer, pl_module, outputs, batch, batch_idx):
        logits = outputs["logits"]
        targets = outputs["targets"]
        preds = torch.argmax(logits.detach(), dim=1)
        self.val_acc.to(preds.device)(preds, targets)
        self.val_acc_class.to(preds.device)(preds, targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        # global
        pl_module.log(
            "val/acc", self.val_acc, on_epoch=True, on_step=False, metric_attribute="val/acc"
        )
        # per class
        class_names = pl_module.hparams.classification_dict.values()
        accuracies = self.val_acc_class.compute()
        self.log_all_class_metrics(accuracies, class_names, "acc", "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        logits = outputs["logits"]
        targets = outputs["targets"]
        preds = torch.argmax(logits.detach(), dim=1)
        self.test_acc.to(preds.device)(preds, targets)
        self.test_acc_class.to(preds.device)(preds, targets)

    def on_test_epoch_end(self, trainer, pl_module):
        # global
        pl_module.log(
            "test/acc", self.test_acc, on_epoch=True, on_step=False, metric_attribute="test/acc"
        )
        # per class
        class_names = pl_module.hparams.classification_dict.values()
        accuracies = self.test_acc_class.compute()
        self.log_all_class_metrics(accuracies, class_names, "acc", "test")

    def log_all_class_metrics(self, metrics, class_names, metric_name, phase: str):
        for value, class_name in zip(metrics, class_names):
            metric_name_for_log = f"{phase}/{metric_name}/{class_name}"
            self.log(
                metric_name_for_log,
                value,
                on_step=False,
                on_epoch=True,
                metric_attribute=metric_name_for_log,
            )

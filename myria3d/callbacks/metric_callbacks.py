from pytorch_lightning import Callback
import torch
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall


class ModelMetrics(Callback):
    """Compute metrics for multiclass classification.

    Accuracy, Precision, Recall are micro-averaged.
    IoU (Jaccard Index) is macro-average to get the mIoU.
    All metrics are also computed per class.

    Be careful when manually computing/reseting metrics. See:
    https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html

    """

    def __init__(self, num_classes=7):
        self.num_classes = num_classes
        self.metrics = {
            "train": self._metrics_factory(),
            "val": self._metrics_factory(),
            "test": self._metrics_factory(),
        }
        self.metrics_by_class = {
            "train": self._metrics_factory(by_class=True),
            "val": self._metrics_factory(by_class=True),
            "test": self._metrics_factory(by_class=True),
        }

    def _metrics_factory(self, by_class=False):
        average = None if by_class else "micro"
        average_iou = None if by_class else "macro"  # special case, only mean IoU is of interest

        return {
            "acc": Accuracy(task="multiclass", num_classes=self.num_classes, average=average),
            "precision": Precision(
                task="multiclass", num_classes=self.num_classes, average=average
            ),
            "recall": Recall(task="multiclass", num_classes=self.num_classes, average=average),
            "f1": F1Score(task="multiclass", num_classes=self.num_classes, average=average),
            # DEBUG: checking that this iou matches the one from model.py before removing it
            "iou-DEV": JaccardIndex(
                task="multiclass", num_classes=self.num_classes, average=average_iou
            ),
        }

    def _end_of_batch(self, phase: str, outputs):
        targets = outputs["targets"]
        preds = torch.argmax(outputs["logits"].detach(), dim=1)
        for m in self.metrics[phase].values():
            m.to(preds.device)(preds, targets)
        for m in self.metrics_by_class[phase].values():
            m.to(preds.device)(preds, targets)

    def _end_of_epoch(self, phase: str, pl_module):
        for metric_name, metric in self.metrics[phase].items():
            metric_name_for_log = f"{phase}/{metric_name}"
            self.log(
                metric_name_for_log,
                metric,
                on_epoch=True,
                on_step=False,
                metric_attribute=metric_name_for_log,
            )
        class_names = pl_module.hparams.classification_dict.values()
        for metric_name, metric in self.metrics_by_class[phase].items():
            values = metric.compute()
            for value, class_name in zip(values, class_names):
                metric_name_for_log = f"{phase}/{metric_name}/{class_name}"
                self.log(
                    metric_name_for_log,
                    value,
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=metric_name_for_log,
                )
            metric.reset()  # always reset when using compute().

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_of_batch("train", outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_of_batch("val", outputs)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_of_batch("test", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        self._end_of_epoch("train", pl_module)

    def on_val_epoch_end(self, trainer, pl_module):
        self._end_of_epoch("val", pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._end_of_epoch("test", pl_module)

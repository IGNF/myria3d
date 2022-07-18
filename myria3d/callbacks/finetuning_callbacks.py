from pytorch_lightning.callbacks import BaseFinetuning
import torch

from myria3d.models.modules.randla_net import SharedMLP
from torch import nn


class FinetuningFreezeUnfreeze(BaseFinetuning):
    def __init__(
        self,
        d_in: int = 9,
        num_classes: int = 6,
        unfreeze_fc_end_epoch: int = 3,
        unfreeze_decoder_epoch: int = 10,
        unfreeze_encoder_epoch: int = 20,
    ):
        super().__init__()

        self._d_in = d_in
        self._num_classes = num_classes
        self._unfreeze_decoder_epoch = unfreeze_decoder_epoch
        self._unfreeze_fc_end_epoch = unfreeze_fc_end_epoch
        self._unfreeze_encoder_epoch = unfreeze_encoder_epoch

    def freeze_before_training(self, pl_module):
        """Update in and out dimensions, and freeze everything at start."""

        # here we could both load the model weights and update its dim afterward
        pl_module.model.change_num_class_for_finetuning(self._num_classes)
        self.freeze(pl_module.model)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        """Unfreeze layers sequentially, starting from the end of the architecture."""
        if current_epoch == 0:
            # final_linear = SharedMLP(128, 5, activation_fn=nn.Sigmoid(), bn=False)
            final_linear = nn.Linear(64, 6, device=pl_module.device)
            # we expect
            # xy around 0 == (0.5-0.5) * 50.0
            # d ~ 6m = (0.12-0.0) * 50.0
            # high value for sigmoid at start.
            p = torch.Tensor([0.47, 0.47, 0.53, 0.53, 0.12, 0.45])
            final_linear.bias = torch.nn.Parameter(
                torch.log(p / (1 - p))
            ).requires_grad_(True)
            pl_module.model.regression_layer = nn.Sequential(
                nn.Linear(512, 256),
                torch.nn.ReLU(),
                nn.Linear(256, 64),
                torch.nn.ReLU(),
                final_linear,
                torch.nn.Sigmoid(),
            ).to(pl_module.device)

        if current_epoch == self._unfreeze_encoder_epoch // 2:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.mlp,
                optimizer=optimizer,
                train_bn=True,
                initial_denom_lr=10,
            )
        if current_epoch == self._unfreeze_encoder_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.encoder,
                optimizer=optimizer,
                train_bn=True,
                initial_denom_lr=100,
            )

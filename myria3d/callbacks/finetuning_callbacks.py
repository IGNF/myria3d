from pytorch_lightning.callbacks import BaseFinetuning


class FinetuningFreezeUnfreeze(BaseFinetuning):
    def __init__(
        self,
        d_in: int = 9,
        num_classes: int = 6,
        unfreeze_fc_end_epoch: int = 3,
        unfreeze_decoder_train_epoch: int = 6,
    ):
        super().__init__()

        self._d_in = d_in
        self._num_classes = num_classes
        self._unfreeze_decoder_epoch = unfreeze_decoder_train_epoch
        self._unfreeze_fc_end_epoch = unfreeze_fc_end_epoch

    def freeze_before_training(self, pl_module):
        """Update in and out dimensions, and freeze everything at start."""

        # here we could both load the model weights and update its dim afterward
        pl_module.model.change_num_class_for_finetuning(self._num_classes)
        self.freeze(pl_module.model)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        """Unfreeze layers sequentially, starting from the end of the architecture."""
        if current_epoch == 0:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.fc_end[-1],
                optimizer=optimizer,
                train_bn=True,
                initial_denom_lr=100,
            )
        if current_epoch == self._unfreeze_fc_end_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.fc_end,
                optimizer=optimizer,
                train_bn=True,
                initial_denom_lr=100,
            )
        if current_epoch == self._unfreeze_decoder_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.decoder,
                optimizer=optimizer,
                train_bn=True,
                initial_denom_lr=100,
            )

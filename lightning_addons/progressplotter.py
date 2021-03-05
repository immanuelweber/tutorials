from collections import defaultdict

import numpy as np
import pytorch_lightning
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import Callback


class ProgressPlotter(Callback):
    def __init__(self, highlight_best=True, show_extra_losses=True):
        self.highlight_best = highlight_best
        self.show_extra_losses = show_extra_losses
        self.metrics = []
        self.train_loss = []
        self.val_loss = []
        self.min_val_loss = np.inf
        self.min_val_loss_step = 0
        self.extra_metrics = defaultdict(list)
        self.steps = []
        self.did = None
        self.is_training = False

    def on_train_start(self, trainer, pl_module) -> None:
        self.is_training = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.train_loss.append(float(trainer.progress_bar_dict["loss"]))

    def on_train_end(self, trainer, pl_module) -> None:
        self.is_training = False

    def on_train_epoch_end(self, trainer, pl_module, outputs) -> None:
        # NOTE: this is due to on_epoch_end currently being called after on_train_epoch_end() 
        # and on_validation_epoch_end()
        if trainer.val_dataloaders is None:
            self.plot(trainer)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # NOTE: this is due to on_epoch_end currently being called after on_train_epoch_end() 
        # and on_validation_epoch_end()
        # since this will only be called if validation dataloaders are available, we need to check
        # if we are in a training cycle to prevent reporting in sanity checking, which also calls on_epoch_end() and on_validation_epoch_end()
        if self.is_training:
            self.plot(trainer)

    # def on_epoch_end(self, trainer, pl_module: pytorch_lightning.LightningModule) -> None:
        # pass

    def plot(self, trainer):

        val_loss = None
        raw_metrics = trainer.logged_metrics.copy()
        raw_metrics.pop("epoch")
        if "loss" in raw_metrics:
            raw_metrics.pop("loss")
        if "val_loss" in raw_metrics:
            val_loss = float(raw_metrics.pop("val_loss"))
        elif "val_loss_epoch" in raw_metrics:
            val_loss = float(raw_metrics.pop("val_loss_epoch"))
        if "val_loss_step" in raw_metrics:
            raw_metrics.pop("val_loss_step")
        if val_loss is not None:
            self.val_loss.append(val_loss)
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.min_val_loss_step = trainer.global_step
        self.steps.append(trainer.global_step)
        for key, value in raw_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu()
            self.extra_metrics[key].append(value)

        fig, ax = plt.subplots()
        plt.close(fig)
        xlim = (trainer.max_epochs * trainer.num_training_batches) or trainer.max_steps
        ax.set_xlim(0, xlim)
        ax.set_xlabel("step")

        ax.plot(self.train_loss, label="loss")
        if val_loss is not None:
            ph = ax.plot(self.steps, self.val_loss, label="val_loss")
            if self.highlight_best:
                ax.plot(
                    self.min_val_loss_step, self.min_val_loss, "o", c=ph[0].get_color()
                )
        lines, labels = ax.get_legend_handles_labels()

        if len(self.extra_metrics) > 0 and self.show_extra_losses:
            extra_ax = ax.twinx()
            extra_ax.set_ylabel("extra losses")
            for key in sorted(self.extra_metrics.keys()):
                extra_ax.plot(self.steps, self.extra_metrics[key], "--", label=key)
            extra_lines, extra_labels = extra_ax.get_legend_handles_labels()
            lines += extra_lines
            labels += extra_labels
        ax.legend(lines, labels, loc=0)

        if self.did:
            self.did.update(fig)
        else:
            self.did = display(fig, display_id=23)

    def static_plot(self, ax=None):
        if ax is None:
            f, ax = plt.subplots()
        ax.plot(self.train_loss, label="loss")
        if self.val_loss:
            ax.plot(self.steps, self.val_loss, label="val_loss")
        lines, labels = ax.get_legend_handles_labels()

        if len(self.extra_metrics) > 0 and self.show_extra_losses:
            extra_ax = ax.twinx()
            extra_ax.set_ylabel("extra losses")
            for key in sorted(self.extra_metrics.keys()):
                extra_ax.plot(self.steps, self.extra_metrics[key], "--", label=key)
            extra_lines, extra_labels = extra_ax.get_legend_handles_labels()
            lines += extra_lines
            labels += extra_labels
        ax.legend(lines, labels, loc=0)

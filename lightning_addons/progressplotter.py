import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

# for multiple y axis see
# https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales


def get_scheduler_names(schedulers):
    names = []
    for scheduler in schedulers:
        sch = scheduler["scheduler"]
        if scheduler["name"] is not None:
            name = scheduler["name"]
        else:
            opt_name = "lr-" + sch.optimizer.__class__.__name__
            i, name = 1, opt_name
            # Multiple schduler of the same type
            while True:
                if name not in names:
                    break
                i, name = i + 1, f"{opt_name}-{i}"

        param_groups = sch.optimizer.param_groups
        if len(param_groups) != 1:
            for i in range(len(param_groups)):
                names.append(f"{name}/pg{i + 1}")
        else:
            names.append(name)
    return names


def get_lrs(schedulers, scheduler_names, interval):
    latest_stat = {}

    for name, scheduler in zip(scheduler_names, schedulers):
        if scheduler["interval"] == interval or interval == "any":
            opt = scheduler["scheduler"].optimizer
            param_groups = opt.param_groups
            for i, pg in enumerate(param_groups):
                suffix = f"/pg{i + 1}" if len(param_groups) > 1 else ""
                lr = {f"{name}{suffix}": pg.get("lr")}
                latest_stat.update(lr)
        else:
            print(f"warning: interval {scheduler['interval']} not supported yet.")

    return latest_stat


class ProgressPlotter(Callback):
    # NOTE: since lightning introduced changes to the callback order on_epoch_* is useless
    # they are called prior and after each dataset cycle of train, val and test
    # this is the reason for the somehow akward use of callbacks
    def __init__(
        self, highlight_best=True, show_extra_losses=True, show_steps=True, show_lr=True
    ):
        self.highlight_best = highlight_best
        self.best_of = "val"  # not implemented
        self.show_extra_losses = show_extra_losses
        self.metrics = []
        self.train_loss = []
        self.val_loss = []
        self.extra_metrics = defaultdict(list)
        self.extra_style = "--"
        self.steps = []
        self.did = None
        self.is_training = False
        self.show_lr = show_lr
        self.lrs = defaultdict(list)
        self.lr_color = plt.cm.viridis(0.5)
        self.show_steps = show_steps

    def on_train_start(self, trainer, pl_module: LightningModule) -> None:
        self.is_training = True
        self.scheduler_names = get_scheduler_names(trainer.lr_schedulers)
        self.steps_per_epoch = trainer.num_training_batches

    def on_train_batch_end(
        self,
        trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.train_loss.append(float(trainer.progress_bar_dict["loss"]))
        lrs = get_lrs(trainer.lr_schedulers, self.scheduler_names, "step")
        for k, v in lrs.items():
            self.lrs[k].append(v)

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        self.is_training = False

    def on_train_epoch_end(
        self, trainer, pl_module: LightningModule, outputs: Any
    ) -> None:
        if trainer.val_dataloaders is None:
            self.collect_metrics(trainer)
            self.update_plot(
                trainer, self.highlight_best, self.show_lr, self.show_steps
            )

    def on_validation_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        if self.is_training:
            self.collect_metrics(trainer)
            self.update_plot(
                trainer, self.highlight_best, self.show_lr, self.show_steps
            )

    def collect_metrics(self, trainer):
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
        self.steps.append(trainer.global_step)
        for key, value in raw_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu()
            self.extra_metrics[key].append(value)

    def update_plot(self, trainer, highlight_best, show_lr, show_steps):
        fig, ax = plt.subplots()
        plt.close(fig)
        if trainer.max_steps:
            max_steps_from_epochs = trainer.max_epochs * trainer.num_training_batches
            max_steps = min(trainer.max_steps, max_steps_from_epochs)
        else:
            max_steps = trainer.max_epochs * trainer.num_training_batches
        self.static_plot(ax, show_lr, highlight_best, show_steps, max_steps=max_steps)

        if self.did:
            self.did.update(fig)
        else:
            rand_id = random.randint(0, 1e6)
            self.did = display(fig, display_id=23 + rand_id)

    def static_plot(
        self,
        ax=None,
        show_lr=True,
        highlight_best=False,
        show_steps=True,
        max_steps=None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        max_steps = max_steps if max_steps else len(self.train_loss)
        step_ax = ax.twiny()
        step_ax.set_xlabel("step")
        step_ax.set_xlim(0, max_steps)
        if not show_steps:
            step_ax.set_xticks([])
            step_ax.set_xlabel("")

        step_ax.plot(self.train_loss, label="loss")
        ax.set_xlabel("epoch")
        ax.set_xlim([0, max_steps / self.steps_per_epoch])

        if self.val_loss:
            ph = step_ax.plot(self.steps, self.val_loss, label="val_loss")
            if highlight_best:
                best_epoch = np.argmin(self.val_loss)
                best_step = (best_epoch + 1) * self.steps_per_epoch
                best_loss = self.val_loss[best_epoch]
                step_ax.plot(best_step, best_loss, "o", c=ph[0].get_color())
        lines, labels = step_ax.get_legend_handles_labels()

        if len(self.extra_metrics) and self.show_extra_losses:
            extra_ax = step_ax.twinx()
            extra_ax.set_ylabel("extra metrics")
            for key in sorted(self.extra_metrics.keys()):
                extra_ax.plot(
                    self.steps, self.extra_metrics[key], self.extra_style, label=key
                )
            extra_lines, extra_labels = extra_ax.get_legend_handles_labels()
            lines += extra_lines
            labels += extra_labels
        if show_lr and len(self.lrs):
            lr_ax = step_ax.twinx()
            lr_ax.set_ylabel("lr")
            lr_ax.spines["right"].set_position(("outward", 60))
            for key, lrs in self.lrs.items():
                lr_ax.plot(lrs, c=self.lr_color, label=key)
            lr_ax.yaxis.label.set_color(self.lr_color)

        step_ax.legend(lines, labels, loc=0)

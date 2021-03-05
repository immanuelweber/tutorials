import time

import pandas as pd
import pytorch_lightning
from IPython.display import display
from pytorch_lightning.callbacks import Callback

# TODO: sort train_*, val_*
# TODO: separate colorings for train_*, val_* changes?


def format_time(t):
    "Format `t` (in seconds) to (h):mm:ss"
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}"


def improvement_styler(df):
    # https://stackoverflow.com/questions/50220200/conditional-styling-in-pandas-using-other-columns
    worse_color = "color: red"
    better_color = "color: black"
    loss_entry = "val_loss" if "val_loss" in df else "loss"
    mask = df[loss_entry].diff() > 0

    # DataFrame with same index and columns names as original filled empty strings
    styled_df = pd.DataFrame(better_color, index=df.index, columns=df.columns)
    styled_df.loc[mask] = worse_color

    min_loss = df[loss_entry].min()
    mask = df[loss_entry] == min_loss
    styled_df.loc[mask] = "font-weight: bold"
    return styled_df


class ProgressPrinter(Callback):
    def __init__(self, highlight_best=True, console=False, log=None):
        self.highlight_best = highlight_best
        self.console = console
        self.log = log
        self.metrics = []
        self.last_time = 0
        self.display_obj = None
        self.is_training = False

    def on_train_start(self, trainer, pl_module) -> None:
        self.is_training = True

    def on_train_end(self, trainer, pl_module) -> None:
        self.is_training = False

    def on_epoch_start(self, trainer, pl_module):
        self.last_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs) -> None:
        # NOTE: this is due to on_epoch_end currently being called after on_train_epoch_end() 
        # and on_validation_epoch_end()
        if trainer.val_dataloaders is None:
            self.report(trainer)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # NOTE: this is due to on_epoch_end currently being called after on_train_epoch_end() 
        # and on_validation_epoch_end()
        # since this will only be called if validation dataloaders are available, we need to check
        # if we are in a training cycle to prevent reporting in sanity checking, which also calls on_epoch_end() and on_validation_epoch_end()
        if self.is_training:
            self.report(trainer)

    # def on_epoch_end(self, trainer, pl_module: pytorch_lightning.LightningModule) -> None:
        # pass

    def report(self, trainer):
        raw_metrics = trainer.logged_metrics.copy()
        metrics = {
            "epoch": int(raw_metrics.pop("epoch")),
            # TODO: no mean loss available for in logged metrics, better way?!
            "loss": float(trainer.progress_bar_dict["loss"]) 
        }
        if "loss" in raw_metrics:
            raw_metrics.pop("loss")
        if "train_loss" in raw_metrics:
            raw_metrics.pop("train_loss")
        if "train_loss_epoch" in raw_metrics:
            raw_metrics.pop("train_loss_epoch")

        if "val_loss" in raw_metrics:
            metrics["val_loss"] = float(raw_metrics.pop("val_loss"))
        elif "val_loss_epoch" in raw_metrics:
            metrics["val_loss"] = float(raw_metrics.pop("val_loss_epoch"))

        for key, value in raw_metrics.items():
            if key.endswith("_epoch"):
                metrics[key[6:]] = float(value)
            elif not key.endswith("_step"):
                metrics[key] = float(value)

        now = time.time()
        elapsed_time = now - self.last_time
        metrics["time"] = format_time(elapsed_time)
        self.metrics.append(metrics)
        metrics_df = pd.DataFrame.from_records(self.metrics)
        if not self.console:
            # https://stackoverflow.com/questions/49239476/hide-a-pandas-column-while-using-style-apply
            if self.highlight_best:
                metrics_df = metrics_df.style.apply(
                    improvement_styler, axis=None
                ).hide_index()
            if not self.display_obj:
                self.display_obj = display(metrics_df, display_id=42)
            else:
                self.display_obj.update(metrics_df)
        else:
            last_row = metrics_df.iloc[-1]

            metrics = {index: last_row[index] for index in last_row.index}
            metrics = {key: f"{val:.4f}" if isinstance(val, float) else val for key, val in metrics.items()}

            metrics = ", ".join(
                [f"{key}: {val}" for key, val in metrics.items() if key != "epoch"]
            )
            pad = len(str(trainer.max_epochs))
            if self.log:
                self.log.info(f"{last_row.name:>{pad}}/{trainer.max_epochs}: {metrics}")
            else:
                print(f"{last_row.name:>{pad}}/{trainer.max_epochs}: {metrics}")

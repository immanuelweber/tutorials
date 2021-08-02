import random
import time

import numpy as np
import pandas as pd
from IPython.display import display
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback


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
    def __init__(
        self, highlight_best: bool = True, console: bool = False, python_logger=None
    ):
        self.highlight_best = highlight_best
        self.console = console
        self.python_logger = python_logger
        self.metrics = []
        self.best_epoch = {"loss": np.inf, "val_loss": np.inf}
        self.last_time = 0
        self.display_obj = None

    def on_train_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        self.last_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module: LightningModule, outputs) -> None:
        self.print(trainer)

    def print(self, trainer) -> None:
        raw_metrics = trainer.logged_metrics.copy()
        metrics = {
            "epoch": trainer.current_epoch,
            # TODO: no mean loss available for in logged metrics, better way?!
            "loss": float(trainer.progress_bar_dict["loss"]),
        }

        ignored_metrics = ["loss", "train_loss", "train_loss_epoch"]
        for m in ignored_metrics:
            if m in raw_metrics:
                raw_metrics.pop(m)

        if "val_loss" in raw_metrics:
            metrics["val_loss"] = float(raw_metrics.pop("val_loss"))
        elif "val_loss_epoch" in raw_metrics:
            metrics["val_loss"] = float(raw_metrics.pop("val_loss_epoch"))

        for key, value in raw_metrics.items():
            if key.endswith("_epoch"):
                metrics[key[6:]] = float(value)
            elif not key.endswith("_step"):
                metrics[key] = float(value)

        if "val_loss" in metrics:
            if metrics["val_loss"] < self.best_epoch["val_loss"]:
                self.best_epoch = metrics
        else:
            if metrics["loss"] < self.best_epoch["loss"]:
                self.best_epoch = metrics

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
                rand_id = random.randint(0, 1e6)
                self.display_obj = display(metrics_df, display_id=42 + rand_id)
            else:
                self.display_obj.update(metrics_df)
        else:
            last_row = metrics_df.iloc[-1]

            metrics = {index: last_row[index] for index in last_row.index}
            metrics = {
                key: f"{val:.4f}" if isinstance(val, float) else val
                for key, val in metrics.items()
            }

            metrics = ", ".join(
                [f"{key}: {val}" for key, val in metrics.items() if key != "epoch"]
            )
            pad = len(str(trainer.max_epochs))
            if self.python_logger:
                self.python_logger.info(
                    f"{last_row.name:>{pad}}/{trainer.max_epochs}: {metrics}"
                )
            else:
                print(f"{last_row.name:>{pad}}/{trainer.max_epochs}: {metrics}")

    def static_print(self, verbose: bool = True) -> pd.DataFrame:
        metrics_df = pd.DataFrame.from_records([self.best_epoch, self.metrics[-1]])
        metrics_df.index = ["best", "last"]
        if verbose:
            display(metrics_df, display_id=43 + random.randint(0, 1e6))
        return metrics_df

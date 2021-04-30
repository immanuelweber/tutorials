from pytorch_lightning.callbacks import ProgressBar
from tqdm.auto import tqdm


class LightProgressBar(ProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)

    def init_predict_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for predicting. """
        bar = tqdm(disable=True)
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(disable=True)
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(disable=True)
        return bar

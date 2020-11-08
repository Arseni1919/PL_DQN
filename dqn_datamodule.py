from CONSTANTS import *


class DQNDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


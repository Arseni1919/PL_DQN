from CONSTANTS import *
from dqn_lightning_module import DQNLightningModule
from dqn_datamodule import DQNDataModule
from dqn_callbaks import DQNCallback
from dqn_dataset import DQNDataset
from try_weights import play


def main():
    dataset = DQNDataset()
    model = DQNLightningModule(dataset)
    data_module = DQNDataModule(dataset)

    trainer = pl.Trainer(callbacks=[DQNCallback()], max_epochs=MAX_EPOCHS)
    trainer.fit(model=model, datamodule=data_module)

    play(NUMBER_OF_GAMES)


if __name__ == '__main__':
    main()

    # to run tensorboard:
    # tensorboard --logdir lightning_logs


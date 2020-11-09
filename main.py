from CONSTANTS import *
from dqn_lightning_module import DQNLightningModule
from dqn_datamodule import DQNDataModule
from dqn_callbaks import DQNCallback
from dqn_dataset import DQNDataset


def main():
    dataset = DQNDataset()
    model = DQNLightningModule(dataset)
    data_module = DQNDataModule(dataset)

    # trainer = pl.Trainer(
    #     gpus=1,
    #     distributed_backend='dp',
    #     max_epochs=500,
    #     early_stop_callback=False,
    #     val_check_interval=100
    # )
    trainer = pl.Trainer(callbacks=[DQNCallback()], max_epochs=200)
    # trainer.save_checkpoint("example.ckpt")

    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()

from CONSTANTS import *
from dqn_lightning_module import DQNLightningModule
from dqn_datamodule import DQNDataModule
from dqn_callbaks import DQNCallback

def main():
    model = DQNLightningModule()
    data_module = DQNDataModule()

    # trainer = pl.Trainer(
    #     gpus=1,
    #     distributed_backend='dp',
    #     max_epochs=500,
    #     early_stop_callback=False,
    #     val_check_interval=100
    # )
    trainer = pl.Trainer(callbacks=[DQNCallback()])

    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()

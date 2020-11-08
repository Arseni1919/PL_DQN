from CONSTANTS import *


def main():
    model = DQNLightning()
    data_loader = DQNDataLoader()

    trainer = pl.Trainer(
        gpus=1,
        # distributed_backend='dp',
        max_epochs=500,
        early_stop_callback=False,
        val_check_interval=100
    )

    trainer.fit(model, data_loader)


if __name__ == '__main__':
    main()

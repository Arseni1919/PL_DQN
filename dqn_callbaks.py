from CONSTANTS import *


class DQNCallback(Callback):

    def on_init_start(self, trainer):
        print('--- Starting to init trainer! ---')

    def on_init_end(self, trainer):
        print('--- trainer is init now ---')

    def on_train_end(self, trainer, pl_module):
        print('--- training ends ---')
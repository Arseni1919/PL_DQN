# My DQN implementation as PL system
The separate parts:
- Data module
- Neural Nets
- PL module
- Callbacks
- Data set

## `LunarLander-v2` parameters:
```
MAX_EPOCHS = 150  # maximum epoch to execute
BATCH_SIZE = 64  # size of the batches
LR = 1e-3  # learning rate
GAMMA = 0.99  # discount factor
SYNC_RATE = 10  # how many frames do we update the target network
REPLAY_SIZE = 20000  # capacity of the replay buffer
WARM_START_STEPS = REPLAY_SIZE  # how many samples do we use to fill our buffer at the start of training
EPS_LAST_FRAME = int(REPLAY_SIZE / BATCH_SIZE * MAX_EPOCHS)  # what frame should epsilon stop decaying
EPS_START = 1  # starting value of epsilon
EPS_END = 0.01  # final value of epsilon
```
DQN net:
```
self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_action),
        )
```

## `CartPole-v0` parameters:
```
MAX_EPOCHS = 300  # maximum epoch to execute
BATCH_SIZE = 64  # size of the batches
LR = 1e-3  # learning rate
GAMMA = 0.99  # discount factor
SYNC_RATE = 10  # how many frames do we update the target network
REPLAY_SIZE = 1000  # capacity of the replay buffer
WARM_START_STEPS = REPLAY_SIZE  # how many samples do we use to fill our buffer at the start of training
EPS_LAST_FRAME = int(REPLAY_SIZE / BATCH_SIZE * MAX_EPOCHS)  # what frame should epsilon stop decaying
EPS_START = 1  # starting value of epsilon
EPS_END = 0.01  # final value of epsilon
```
DQN net:
```
self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_action),
        )
```
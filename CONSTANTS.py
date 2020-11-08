import numpy as np
import pytorch_lightning as pl
import torch
import os
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning.callbacks import Callback
import gym

BATCH_SIZE = 16  # size of the batches
LR = 1e-2  # learning rate
ENV = "CartPole-v0"  # gym environment tag
GAMMA = 0.99  # discount factor
SYNC_RATE = 10  # how many frames do we update the target network
REPLAY_SIZE = 1000  # capacity of the replay buffer
WARM_START_STEPS = 1000  # how many samples do we use to fill our buffer at the start of training
EPS_LAST_FRAME = 1000  # what frame should epsilon stop decaying
EPS_START = 1  # starting value of epsilon
EPS_END = 0.01  # final value of epsilon
EPISODE_LENGTH = 200  # max length of an episode
MAX_EPISODE_REWARD = 200  # max episode reward in the environment


import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import gzip
import itertools

device = torch.device('cuda')

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

class DynamicsModel(nn.Module):
    def __init__(self, feature_size, hidden_size=256, use_decay=False):
        super(DynamicsModel, self).__init__()
        self.hidden_size = hidden_size
        # self.nn1 = nn.Linear(feature_size + action_size, hidden_size)
        # self.nn2 = nn.Linear(hidden_size, feature_size + reward_size)
        self.nn1 = nn.Linear(feature_size + feature_size, feature_size)
        self.use_decay = use_decay
        self.apply(weights_init_)
        self.swish = Swish()

    def forward(self, state_latent, action_latent):
        x = torch.cat([state_latent, action_latent], 1)
        # nn1_output = self.swish(self.nn1(x))
        # nn2_output = self.swish(self.nn2(nn1_output))

        nn1_output = self.nn1(x)
        return nn1_output

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if type(m) == nn.Linear:
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss


class RewardModel(nn.Module):
    def __init__(self, feature_size, reward_size, hidden_size=256, use_decay=False):
        super(RewardModel, self).__init__()
        self.hidden_size = hidden_size
        # self.nn1 = nn.Linear(feature_size + action_size, hidden_size)
        # self.nn2 = nn.Linear(hidden_size, feature_size + reward_size)
        self.nn1 = nn.Linear(feature_size + feature_size, reward_size)
        self.use_decay = use_decay
        self.apply(weights_init_)
        self.swish = Swish()

    def forward(self, state_latent, action_latent):
        x = torch.cat([state_latent, action_latent], 1)
        # nn1_output = self.swish(self.nn1(x))
        # nn2_output = self.swish(self.nn2(nn1_output))

        nn1_output = self.nn1(x)
        return nn1_output

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if type(m) == nn.Linear:
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss


import random

import numpy as np
import torch
from torch.nn import Module, Linear, ReLU, Sequential, init

from torch.distributions import Normal

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

mse_loss_function = torch.nn.MSELoss()

torch.autograd.set_detect_anomaly(True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        try:
            print("Initializing")
            init.uniform_(m.weight.data, -0.003, 0.003)
            init.uniform_(m.bias.data, -0.003, 0.003)
            # m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class Actor(Module):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        # for cts critic, it can only outout Q(S,A), so it needs both action and state
        self.lin1 = Sequential(Linear(in_features=n_states, out_features=500), ReLU())
        self.lin2 = Sequential(Linear(in_features=500, out_features=500), ReLU())
        self.mu = Sequential(Linear(in_features=500, out_features=n_actions))
        self.logstd = Sequential(Linear(in_features=500, out_features=n_actions))

        # self.apply(weights_init)

        # self.mu.weight.data.uniform_(-0.003, 0.003)
        # self.mu.weight.bias.data.uniform(-0.003, 0.003)
        # self.logstd.weight.data.uniform_(-0.003, 0.003)
        # self.logstd.weight.bias.data.uniform(-0.003, 0.003)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        mu = self.mu(x)
        logstd = self.logstd(x)
        logstd = torch.clamp(logstd, -20, 2)
        return mu, logstd

    def get_action(self, state, train):
        mean, logstd = self.forward(state)  # (batch, n_actions*2)
        std = logstd.exp()
        dist = Normal(mean, std)

        u = dist.rsample()  # (batch, n_actions)
        # to bound the action within [-1, 1]
        # This was used in the paper, but it also matches with LunarLander's action bound as well
        sampled_action = torch.tanh(u)

        # sum is there to get the actual probability of this random variable
        # All of the dimensions are treated as independent variables
        # Therefore multipliying probability of each values in the vector will result in total sum
        # However, since this is the log probability, instead of multiplying, you would add instead
        # mu_log_prob = torch.sum(dist.log_prob(u), 1, keepdim=True)  # log prob of mu(u|s)
        log_pi = torch.sum(dist.log_prob(u), 1, keepdim=True) - torch.sum(torch.log(1 - sampled_action.pow(2) + 1e-6),
                                                                          1, keepdim=True)

        # log_pi = dist.log_prob(u) - torch.log(1 - sampled_action.pow(2) + 1e-6)

        # log_pi = log_pi.sum(1, keepdim=True)
        if train:
            return sampled_action, log_pi
        else:
            return torch.tanh(mean).detach().cpu().numpy().squeeze(), log_pi


class Critic(Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        total_input_size = n_states + n_actions
        self.lin1 = Sequential(Linear(in_features=total_input_size, out_features=500), ReLU())
        self.lin2 = Sequential(Linear(in_features=500, out_features=500), ReLU())
        # for each action, you produce corresponding mean and variance
        self.final_lin = Linear(in_features=500, out_features=1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        x = self.lin1(x)
        x = self.lin2(x)
        output = self.final_lin(x)
        return output

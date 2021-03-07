import argparse
import os
import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
import tensorflow as tf
from torch.nn import Module, Linear, ReLU, Sequential, Softmax, Parameter
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from torch.optim import Adam
from torch.distributions import Normal

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

mse_loss_function = torch.nn.MSELoss()

torch.autograd.set_detect_anomaly(True)
class Actor(Module):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        # for cts critic, it can only outout Q(S,A), so it needs both action and state
        self.lin1 = Sequential(Linear(in_features=n_states, out_features=64), ReLU())
        # self.lin2 = Sequential(Linear(in_features=16, out_features=24), ReLU())
        self.mu = Sequential(Linear(in_features=64, out_features=n_actions))
        self.logstd = Sequential(Linear(in_features=64, out_features=n_actions))

    def forward(self, x):
        x = self.lin1(x)
        # x = self.lin2(x)
        mu = self.mu(x)
        log_std = self.mu(x)
        return mu, log_std


class Critic(Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        total_input_size = n_states + n_actions
        self.lin1 = Sequential(Linear(in_features=total_input_size, out_features=64), ReLU())
        # self.lin2 = Sequential(Linear(in_features=150, out_features=150), ReLU())
        # for each action, you produce corresponding mean and variance
        self.final_lin = Sequential(Linear(in_features=64, out_features=1))

    def forward(self, x):
        x = self.lin1(x)
        # x = self.lin2(x)
        # x = self.lin3(x)
        # print(x)
        output = self.final_lin(x)
        return output


class SAC:
    def __init__(self, n_states, n_actions):
        self.replay_size = 100000
        self.experience_replay = deque(maxlen=self.replay_size)
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.0003
        self.batch_size = 128
        self.gamma = 0.999
        self.actor = Actor(n_states=n_states, n_actions=n_actions)
        self.critic = Critic(n_states=n_states, n_actions=n_actions)
        self.target_critic = Critic(n_states=n_states, n_actions=n_actions)
        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)
        self.optim_critic = Adam(params=self.critic.parameters(), lr=self.lr)
        # self.H = 0.98 * (-np.log(1 / self.n_actions))
        self.H = -2
        self.Tau = 0.005
        self.alpha = Parameter(torch.tensor(0.5))
        self.optim_alpha = Adam(params=[self.alpha], lr=self.lr)

    def get_action(self, state, test=False):
        mean, logstd = self.actor(state.float())  # (batch, n_actions*2)
        # mean, logstd = mean_and_logvar[:, 0:self.n_actions], mean_and_logvar[:,
        #                                                      self.n_actions:]  # (batch, n_actions*2) each
        logstd = torch.clamp(logstd, -20, 2)
        std = torch.exp(logstd)

        dist = Normal(mean, std)
        u = dist.rsample()  # (batch, n_actions)
        # to bound the action within [-1, 1]
        # This was used in the paper, but it also matches with LunarLander's action bound as well
        # print("u",u)
        if (torch.isnan(u)).any():
            for name, param in self.actor.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
            print(state)
            print(f"mean:{mean}, logstd:{logstd}, std:{std}")
            exit(0)
        sampled_action = torch.tanh(u)

        # sum is there to get the actual probability of this random variable
        # All of the dimensions are treated as independent variables
        # Therefore multipliying probability of each values in the vector will result in total sum
        # However, since this is the log probability, instead of multiplying, you would add instead
        # mu_log_prob = torch.sum(dist.log_prob(u), 1, keepdim=True)  # log prob of mu(u|s)
        mu_log_prob = dist.log_prob(u)
        pi_log_prob = torch.sum((mu_log_prob - torch.log(1 - sampled_action.pow(2) + 0.0001)), dim=1, keepdim=True)  # (batch, 1)
        if not test:
            return sampled_action, pi_log_prob
        else:
            return torch.tanh(mean).detach().cpu().numpy().squeeze(), pi_log_prob

    def get_v(self, state_batch):
        sample_action, log_action_probs = self.get_action(state=state_batch, test=False)
        sample_action = sample_action.detach()  # (batch, n_actions)
        log_action_probs = log_action_probs.detach()  # (batch, 1)
        s_a_pair = torch.cat([state_batch, sample_action], dim=1)
        q_values = self.target_critic(s_a_pair).detach()  # (batch, 1)
        value = q_values - self.alpha * log_action_probs
        return value  # (batch, 1)

    def train_critic(self, s_currs, a_currs, r, s_nexts, dones):
        # using equation (5) from the second paper
        self.optim_critic.zero_grad()

        s_a_pair = torch.cat([s_currs, a_currs], dim=1)
        predicts = self.critic(s_a_pair)  # (batch, actions)
        v_vector = self.get_v(s_nexts)

        target = r + self.gamma * v_vector  # (batch, 1)
        done_indices = np.argwhere(dones)
        if done_indices.shape[1] > 0:
            done_indices = torch.squeeze(dones.nonzero())
            target[done_indices, 0] = torch.squeeze(r[done_indices])

        loss = mse_loss_function(predicts, target)
        loss.backward()
        # print(loss)
        self.optim_critic.step()
        return

    # actor -> policy network (improve policy network)
    def train_actor(self, s_currs):
        # s_a_pair = torch.cat([s_currs, a_currs], dim=1)
        self.optim_actor.zero_grad()
        sample_action, log_action_probs = self.get_action(state=s_currs, test=False)
        s_a_pair = torch.cat([s_currs, sample_action], dim=1)
        q_values = self.critic(s_a_pair).detach()
        loss = (self.alpha * log_action_probs) - q_values
        loss = torch.mean(loss)
        # print(loss)
        loss.backward()
        self.optim_actor.step()

    def train_alpha(self, s_currs):
        self.optim_alpha.zero_grad()
        sample_action, log_action_probs = self.get_action(state=s_currs, test=False)
        loss = -1 * self.alpha * (log_action_probs + self.H)
        loss = torch.mean(loss)
        # print(loss)
        loss.backward()

        self.optim_alpha.step()

    def process_batch(self, x_batch):
        s_currs = torch.zeros((self.batch_size, self.n_states))
        a_currs = torch.zeros((self.batch_size, self.n_actions))
        r = torch.zeros((self.batch_size, 1))
        s_nexts = torch.zeros((self.batch_size, self.n_states))
        dones = torch.zeros((self.batch_size,))

        for batch in range(self.batch_size):
            s_currs[batch] = x_batch[batch].s_curr
            a_currs[batch] = x_batch[batch].a_curr
            r[batch] = x_batch[batch].reward
            s_nexts[batch] = x_batch[batch].s_next
            dones[batch] = x_batch[batch].done

        return s_currs, a_currs, r, s_nexts, dones

    def train(self, x_batch):
        s_currs, a_currs, r, s_nexts, dones = self.process_batch(x_batch=x_batch)
        self.train_critic(s_currs, a_currs, r, s_nexts, dones)
        self.train_actor(s_currs)
        self.train_alpha(s_currs)
        self.update_weights()
        return

    def update_weights(self):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)


def main(episodes, exp_name):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    # writer = tf.summary.create_file_writer(logdir)
    env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('MountainCarContinuous-v0')
    n_states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.shape[0]
    agent = SAC(n_states=n_states, n_actions=n_actions)
    warmup_ep = 0
    exploration_eps = 100
    for ep in range(episodes):
        s_curr = env.reset()
        s_curr = np.reshape(s_curr, (1, n_states))
        s_curr = s_curr.astype(np.float32)
        done = False
        score = 0
        agent.update_weights()  # update weight every time an episode ends
        step = 0
        while not done:
            # env.render()
            # if len(agent.experience_replay) == agent.replay_size:
            #     env.render()
            s_curr_tensor = torch.from_numpy(s_curr)
            a_curr_tensor, _ = agent.get_action(s_curr_tensor)
            # print(a_curr_tensor)
            # this detach is necessary as the action tensor gets concatenated with state tensor when passed in to critic
            # without this detach, each action tensor keeps its graph, and when same action gets sampled from buffer,
            # it considers that graph "already processed" so it will throw an error
            a_curr_tensor = a_curr_tensor.detach()
            s_next, r, done, _ = env.step(a_curr_tensor.cpu().numpy().flatten())
            # s_next = s_next.flatten() # for mountain car
            s_next_tensor = torch.from_numpy(s_next)
            s_next = np.reshape(s_next, (1, n_states))
            sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])

            # must re-make training dataloader since the dataset is now updated with aggregation of new data

            sample.s_curr = s_curr_tensor
            sample.a_curr = a_curr_tensor
            sample.reward = r
            sample.s_next = s_next_tensor
            sample.done = done

            if len(agent.experience_replay) < agent.replay_size:
                agent.experience_replay.append(sample)
                s_curr = s_next
                print("appending to buffer....")
                continue
            else:
                agent.experience_replay.append(sample)
                # # if step+1 % 100 == 0:
                # #     for _ in range(4000):
                # x_batch = random.sample(agent.experience_replay, agent.batch_size)
                # agent.train(x_batch)
                s_curr = s_next
            score += r
            step += 1

            if done:
                if r >= 100 :
                    print("Landed Successfully")
                elif r == -100:
                    print("Landed Unsuccessfully")
                print(f"ep:{ep - warmup_ep}:################Goal Reached###################", score)
                # with writer.as_default():
                #     tf.summary.scalar("reward", r, ep)
                #     tf.summary.scalar("score", score, ep)
            if step % 1000 == 0:
                print("Training")
                for j in range(1000):
                    x_batch = random.sample(agent.experience_replay, agent.batch_size)
                    agent.train(x_batch)
    return agent


def env_with_render(agent):
    done = False
    env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('MountainCarContinuous-v0')
    score = 0
    states = env.observation_space.shape[0]  # shape returns a tuple
    s_curr = np.reshape(env.reset(), (1, states))
    while True:
        if done:
            print(score)
            score = 0
            s_curr = np.reshape(env.reset(), (1, states))
        env.render()
        s_curr_tensor = torch.from_numpy(s_curr)
        a_curr, _ = agent.get_action(s_curr_tensor, test=True)
        print(a_curr)
        s_next, r, done, _ = env.step(a_curr)
        s_next = np.reshape(s_next, (1, states))
        s_curr = s_next
        score += r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="SAC", help="exp_name")
    ap.add_argument("--episodes", type=int, default=2000*1000, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"])
    env_with_render(agent=trained_agent)

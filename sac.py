import argparse
import os
import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
from torch.distributions import Normal
from torch.optim import Adam
from torch.nn import Parameter

from network import Actor, Critic

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

mse_loss_function = torch.nn.MSELoss()

torch.autograd.set_detect_anomaly(True)


class SAC:
    def __init__(self, n_states, n_actions):
        self.replay_size = 1000000
        self.experience_replay = deque(maxlen=self.replay_size)
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.0003
        self.batch_size = 128
        self.gamma = 0.99
        self.actor = Actor(n_states=n_states, n_actions=n_actions)
        self.critic = Critic(n_states=n_states, n_actions=n_actions)
        self.critic2 = Critic(n_states=n_states, n_actions=n_actions)

        self.target_critic = Critic(n_states=n_states, n_actions=n_actions)
        self.target_critic2 = Critic(n_states=n_states, n_actions=n_actions)
        self.H = -2
        self.Tau = 0.01
        # self.alpha = 0.2
        self.log_alpha = Parameter(torch.tensor(0.0))
        self.optim_alpha = Adam(params=[self.log_alpha], lr=self.lr)
        self.alpha = 0.2

        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(local_param)

        for target_param, local_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(local_param)

        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)
        self.optim_critic = Adam(params=self.critic.parameters(), lr=self.lr)
        self.optim_critic_2 = Adam(params=self.critic2.parameters(), lr=self.lr)

    # def get_action(self, state, test=False):
    #     mean, logstd = self.actor(state.float())  # (batch, n_actions*2)
    #     # mean, logstd = mean_and_logvar[:, 0:self.n_actions], mean_and_logvar[:,
    #     #                                                      self.n_actions:]  # (batch, n_actions*2) each
    #     logstd = torch.clamp(logstd, -20, 2)
    #     std = torch.exp(logstd)
    #
    #     dist = Normal(mean, std)
    #     u = dist.rsample()  # (batch, n_actions)
    #     # to bound the action within [-1, 1]
    #     # This was used in the paper, but it also matches with LunarLander's action bound as well
    #     # print("u",u)
    #     if (torch.isnan(u)).any():
    #         for name, param in self.actor.named_parameters():
    #             if param.requires_grad:
    #                 print(name, param.data)
    #         print(state)
    #         print(f"mean:{mean}, logstd:{logstd}, std:{std}")
    #         exit(0)
    #     sampled_action = torch.tanh(u)
    #
    #     # sum is there to get the actual probability of this random variable
    #     # All of the dimensions are treated as independent variables
    #     # Therefore multipliying probability of each values in the vector will result in total sum
    #     # However, since this is the log probability, instead of multiplying, you would add instead
    #     # mu_log_prob = torch.sum(dist.log_prob(u), 1, keepdim=True)  # log prob of mu(u|s)
    #     mu_log_prob = dist.log_prob(u)
    #     pi_log_prob = torch.sum((mu_log_prob - torch.log(1 - sampled_action.pow(2) + 0.0001)), dim=1,
    #                             keepdim=True)  # (batch, 1)
    #     if not test:
    #         return sampled_action, pi_log_prob
    #     else:
    #         return torch.tanh(mean).detach().cpu().numpy().squeeze(), pi_log_prob

    def get_v(self, state_batch, action, log_action_probs):
        action = action.detach()  # (batch, n_actions)
        log_action_probs = log_action_probs.detach()  # (batch, 1)
        s_a_pair = torch.cat([state_batch, action], dim=1)
        q_values = self.target_critic(s_a_pair).detach()  # (batch, 1)
        q_values_2 = self.target_critic2(s_a_pair).detach()
        # print(q_values, q_values_2)
        value = torch.min(q_values, q_values_2) - self.alpha * log_action_probs
        return value.detach()  # (batch, 1)

    # actor -> policy network (improve policy network)
    def train_actor(self, s_currs):
        sample_action, log_action_probs = self.actor.get_action(state=s_currs, train=True)
        s_a_pair = torch.cat([s_currs, sample_action], dim=1)
        q_values = self.critic(state).detach()
        q_values_2 = self.critic2(s_a_pair).detach()
        loss = (self.alpha * log_action_probs) - torch.min(q_values, q_values_2)
        # print(self.alpha, log_action_probs.mean())
        # print(torch.min(q_values, q_values_2).mean())
        loss = torch.mean(loss)
        self.optim_actor.zero_grad()
        # print(loss)
        loss.backward()
        self.optim_actor.step()

        alpha_loss = torch.mean(-1 * self.log_alpha * (log_action_probs.detach() + self.H))
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()
        self.alpha = torch.exp(self.log_alpha)

    def train_alpha(self, s_currs, log_action_probs):
        self.optim_alpha.zero_grad()
        sample_action, log_action_probs = self.actor.get_action(state=s_currs, train=True)
        loss = -1 * self.log_alpha * (log_action_probs + self.H)
        loss = torch.mean(loss)
        # print(loss)
        loss.backward()

        self.optim_alpha.step()

    def process_batch(self, x_batch):
        s_currs = torch.zeros((self.batch_size, self.n_states))
        a_currs = torch.zeros((self.batch_size, self.n_actions))
        r = torch.zeros((self.batch_size, 1))
        s_nexts = torch.zeros((self.batch_size, self.n_states))
        dones = torch.zeros((self.batch_size, 1))

        for batch in range(self.batch_size):
            s_currs[batch] = x_batch[batch].s_curr
            a_currs[batch] = x_batch[batch].a_curr
            r[batch] = x_batch[batch].reward
            s_nexts[batch] = x_batch[batch].s_next
            dones[batch] = x_batch[batch].done
        dones = dones.float()

        return s_currs, a_currs, r, s_nexts, dones

    def train_critic(self, s_currs, a_currs, r, s_nexts, dones, a_nexts, log_action_probs_next):
        # using equation (5) from the second paper
        s_a_pair = torch.cat([s_currs, a_currs], dim=1)
        predicts = self.critic(s_a_pair)  # (batch, actions)
        predicts2 = self.critic2(s_a_pair)

        v_vector = self.get_v(s_nexts, a_nexts, log_action_probs_next)

        target = r + (1 - dones) * self.gamma * v_vector
        loss = mse_loss_function(predicts, target)
        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()

        loss2 = mse_loss_function(predicts2, target)
        self.optim_critic_2.zero_grad()
        loss2.backward()
        self.optim_critic_2.step()
        return

    def train(self, x_batch, step):
        s_currs, a_currs, r, s_nexts, dones = self.process_batch(x_batch=x_batch)

        a_nexts, log_action_probs_next = self.actor.get_action(s_nexts, train=True)

        predicts = self.critic(s_currs, a_currs)  # (batch, actions)
        predicts2 = self.critic2(s_currs, a_currs)

        # s_a_pair = torch.cat([s_nexts, a_nexts.detach()], dim=1)
        q_values = self.target_critic(s_nexts, a_nexts).detach()  # (batch, 1)
        q_values_2 = self.target_critic2(s_nexts, a_nexts).detach()
        # print(q_values, q_values_2)
        value = torch.min(q_values, q_values_2) - self.alpha * log_action_probs_next

        target = r + ((1 - dones) * self.gamma * value.detach())
        loss = mse_loss_function(predicts, target)
        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()

        loss2 = mse_loss_function(predicts2, target)
        self.optim_critic_2.zero_grad()
        loss2.backward()
        self.optim_critic_2.step()

        sample_action, log_action_probs = self.actor.get_action(state=s_currs, train=True)
        q_values_new = self.critic(s_currs, sample_action)
        q_values_new_2 = self.critic2(s_currs, sample_action)
        loss_actor = (self.alpha * log_action_probs) - torch.min(q_values_new, q_values_new_2)

        loss_actor = torch.mean(loss_actor)
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        alpha_loss = torch.mean((-1 * torch.exp(self.log_alpha)) * (log_action_probs.detach()+ self.H))
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()
        self.alpha = torch.exp(self.log_alpha)
        self.update_weights()
        return

    def update_weights(self):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)

        for target_param, local_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
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
    exploration_eps = -1
    for ep in range(episodes):
        s_curr = env.reset()
        s_curr = np.reshape(s_curr, (1, n_states))
        s_curr = s_curr.astype(np.float32)
        done = False
        score = 0
        step = 0
        while not done:
            s_curr_tensor = torch.from_numpy(s_curr)
            if ep < exploration_eps:
                print("Exploring")
                a_curr = env.action_space.sample()
                a_curr_tensor = torch.from_numpy(a_curr).unsqueeze(0)
            else:
                a_curr_tensor, _ = agent.actor.get_action(s_curr_tensor, train=True)
                # this detach is necessary as the action tensor gets concatenated with state tensor when passed in to critic
                # without this detach, each action tensor keeps its graph, and when same action gets sampled from buffer,
                # it considers that graph "already processed" so it will throw an error
                a_curr_tensor = a_curr_tensor.detach()
                a_curr = a_curr_tensor.cpu().numpy().flatten()

            s_next, r, done, _ = env.step(a_curr)
            # env.render()
            # print(a_curr)
            # print(r, done)
            # s_next = s_next.flatten() # for mountain car

            s_next = np.reshape(s_next, (1, n_states))
            s_next_tensor = torch.from_numpy(s_next)
            sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])
            if step == 500:
                print("RAN FOR TOO LONG")
                done = True
            # must re-make training dataloader since the dataset is now updated with aggregation of new data

            sample.s_curr = s_curr_tensor
            sample.a_curr = a_curr_tensor
            sample.reward = r
            sample.s_next = s_next_tensor
            sample.done = done

            if len(agent.experience_replay) < agent.batch_size:
                agent.experience_replay.append(sample)
                print("appending to buffer....")
            else:
                agent.experience_replay.append(sample)
                if ep > exploration_eps:
                    x_batch = random.sample(agent.experience_replay, agent.batch_size)
                    agent.train(x_batch, step)

            s_curr = s_next
            score += r
            step += 1
            if done:
                print(f"ep:{ep}:################Goal Reached###################", score)
                # with writer.as_default():
                #     tf.summary.scalar("reward", r, ep)
                #     tf.summary.scalar("score", score, ep)
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
    ap.add_argument("--episodes", type=int, default=2000, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"])
    env_with_render(agent=trained_agent)

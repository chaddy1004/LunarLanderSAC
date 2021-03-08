import argparse
import os
import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Parameter
import tensorflow as tf
from network import Actor, Critic

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

mse_loss_function = torch.nn.MSELoss()

torch.autograd.set_detect_anomaly(True)

if torch.cuda.device_count() > 0:
    print("RUNNING ON GPU")
    DEVICE = torch.device('cuda')
else:
    print("RUNNING ON CPU")
    DEVICE = torch.device('cpu')


class SAC:
    def __init__(self, n_states, n_actions):
        # hyper parameters
        self.replay_size = 1000000
        self.experience_replay = deque(maxlen=self.replay_size)
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.0003
        self.batch_size = 128
        self.gamma = 0.99
        self.H = -2
        self.Tau = 0.01

        # actor network
        self.actor = Actor(n_states=n_states, n_actions=n_actions).to(DEVICE)

        # dual critic network, with corresponding targets
        self.critic = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)
        self.critic2 = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)
        self.target_critic = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)
        self.target_critic2 = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)

        # make the target critics start off same as the main networks
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(local_param)

        for target_param, local_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(local_param)

        # temperature variable
        self.log_alpha = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        self.optim_alpha = Adam(params=[self.log_alpha], lr=self.lr)
        self.alpha = 0.2

        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)
        self.optim_critic = Adam(params=self.critic.parameters(), lr=self.lr)
        self.optim_critic_2 = Adam(params=self.critic2.parameters(), lr=self.lr)

    def get_v(self, state_batch, action, log_action_probs):
        # TODO: move code from main train() function
        return

    def train_actor(self, s_currs):
        # TODO: move code from main train() function
        return

    def train_alpha(self, s_currs, log_action_probs):
        # TODO: move code from main train() function
        return

    def train_critic(self, s_currs, a_currs, r, s_nexts, dones, a_nexts, log_action_probs_next):
        # TODO: move code from main train() function
        return

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
        return s_currs.to(DEVICE), a_currs.to(DEVICE), r.to(DEVICE), s_nexts.to(DEVICE), dones.to(DEVICE)

    def train(self, x_batch):
        s_currs, a_currs, r, s_nexts, dones = self.process_batch(x_batch=x_batch)

        a_nexts, log_action_probs_next = self.actor.get_action(s_nexts, train=True)

        predicts = self.critic(s_currs, a_currs)  # (batch, actions)
        predicts2 = self.critic2(s_currs, a_currs)

        q_values = self.target_critic(s_nexts, a_nexts).detach()  # (batch, 1)
        q_values_2 = self.target_critic2(s_nexts, a_nexts).detach()
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

        alpha_loss = torch.mean((-1 * torch.exp(self.log_alpha)) * (log_action_probs.detach() + self.H))
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
    writer = tf.summary.create_file_writer(logdir)
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
                a_curr_tensor, _ = agent.actor.get_action(s_curr_tensor.to(DEVICE), train=True)
                # this detach is necessary as the action tensor gets concatenated with state tensor when passed in to critic
                # without this detach, each action tensor keeps its graph, and when same action gets sampled from buffer,
                # it considers that graph "already processed" so it will throw an error
                a_curr_tensor = a_curr_tensor.detach()
                a_curr = a_curr_tensor.cpu().numpy().flatten()

            s_next, r, done, _ = env.step(a_curr)

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
                    agent.train(x_batch)

            s_curr = s_next
            score += r
            step += 1
            if done:
                print(f"ep:{ep}:################Goal Reached###################", score)
                with writer.as_default():
                    tf.summary.scalar("reward", r, ep)
                    tf.summary.scalar("score", score, ep)
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="SAC_LunarLander_Score", help="exp_name")
    ap.add_argument("--episodes", type=int, default=700, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"])
    if DEVICE == torch.device('cpu'):
        torch.save(trained_agent.actor, "policy_trained_on_cpu.pt")
    else:
        torch.save(trained_agent.actor, "policy_trained_on_gpu.pt")

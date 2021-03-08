import argparse
import random

import gym
import numpy as np
import torch

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

torch.autograd.set_detect_anomaly(True)

if torch.cuda.device_count() > 0:
    print("RUNNING ON GPU")
    DEVICE = torch.device('cuda')
else:
    print("RUNNING ON CPU")
    DEVICE = torch.device('cpu')


def env_with_render(policy):
    done = False
    env = gym.make('LunarLanderContinuous-v2')
    score = 0
    states = env.observation_space.shape[0]  # shape returns a tuple
    s_curr = np.reshape(env.reset(), (1, states))
    while True:
        if done:
            print(score)
            score = 0
            s_curr = np.reshape(env.reset(), (1, states))
        env.rernder()
        s_curr_tensor = torch.from_numpy(s_curr)
        a_curr, _ = policy.get_action(s_curr_tensor, train=False)
        s_next, r, done, _ = env.step(a_curr)
        s_next = np.reshape(s_next, (1, states))
        s_curr = s_next
        score += r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    expert_policy = torch.load("policy_trained_on_gpu.pt", map_location=torch.device('cpu'))
    env_with_render(policy=expert_policy)

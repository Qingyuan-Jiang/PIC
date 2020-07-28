import argparse
import math
from collections import namedtuple
from itertools import count
import numpy as np
from eval import eval_model_q
import copy
import torch
from ddpg_vec import DDPG
from ddpg_vec_hetero import DDPGH
import random

from replay_memory import ReplayMemory, Transition
from utils import *
import os
import time
from utils import n_actions, copy_actor_policy
from ddpg_vec import hard_update
import torch.multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value
import sys


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # env = make_env('simple_spread_n6', None)
    # env = make_env('simple_coop_push_n15', None)
    env = make_env('simple_tag_n3', None)
    # env = make_env('hetero_spread_n4', None)
    n_agents = env.n
    seed = 0
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_adversary = 0

    obs_dims = [env.observation_space[i].shape[0] for i in range(n_agents)]
    obs_dims.insert(0, 0)

    while True:
        env.reset()
        print('Restart')
        for idx in range(100):
            # next_state, rewards, done, reward_info = env.step(np.random.rand(n_agents,5).tolist())
            next_state, rewards, done, reward_info = env.step(np.random.randint(0, 5, (n_agents, 5)).tolist())
            print(rewards)
            env.render()


if __name__ == "__main__":
    main()

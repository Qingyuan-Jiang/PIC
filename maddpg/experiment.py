import random

import numpy as np
import torch
from utils import *


def main(env, agent, seed=0):
    torch.set_num_threads(1)

    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for _ in range(20):
        obs_n = env.reset()
        print('Restart')
        for idx in range(100):
            action_n = agent.select_action(torch.Tensor(obs_n), action_noise=True, param_noise=False)
            action_n = action_n.squeeze().cpu().numpy()
            next_obs_n, reward_n, done_n, _ = env.step(action_n)
            obs_n = next_obs_n
            env.render()


if __name__ == '__main__':

    print("Evaluating the scenario. ")
    fname = 'ckpt_plot/comp_pe_n3/agents.ckpt'
    scenario = 'simple_tag_n3'
    device = torch.device('cpu')

    env = make_env(scenario, None)

    checkpoint = torch.load(fname, map_location=device)
    agent = checkpoint['agents']

    main(env, agent)

    print("Evaluation end. ")

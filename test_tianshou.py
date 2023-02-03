import pickle, numpy as np
import tianshou as ts
from tianshou.data import Batch, ReplayBuffer
buf = ReplayBuffer(size=10)
import torch
from torch import nn

import random
import os

def set_seed(seed = 0):
    # Set seed for result reproducibility
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_obs(obs,max_state=29,sp=0):
    res = torch.zeros(1, max_state + 1)

    if sp==1:
        res[0][obs] = -1
    else:
        res[0][obs] = 1


    return res


for i in range(20):
    tmp = Batch(obs=encode_obs(obs=i), act=i, rew=0.1*i, done=0, obs_next=encode_obs(obs=i+5,sp=1), info={},terminated = 0,truncated=0)
        #Batch(obs=i, act=i, rew=0.1*i, done=0, obs_next=encode_obs(obs=i+5,sp=1), info={},terminated = 0,truncated=0)
    #Batch(obs=encode_obs(obs=i), act=i, rew=0.1*i, done=0, obs_next=encode_obs(obs=i+5,sp=1), info={},terminated = 0,truncated=0)
    buf.add(tmp)

print(buf.obs)



class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        #print(obs)
        idx= np.abs(obs.numpy())
        idx=np.argwhere(idx==1)
        for id in idx:
            print(id[-1])
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

state_shape =[30] #env.observation_space.shape or env.observation_space.n
action_shape =[20]  #env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

set_seed(8)

policy = ts.policy.DQNPolicy(net, optim, discount_factor=0, estimation_step=3, target_update_freq=320) #no discount factor
#ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

policy.set_eps(0.1)

losses = policy.update(2, buf)

batch = Batch(obs=encode_obs(obs=3),info= {},terminated = 0,truncated=0)  # the first dimension is batch-size
act = policy(batch).act[0]  # policy.forward return a batch, use ".act" to extract the action


print(policy)

# for i in range(int(1e6)):  # total step
#     # once if the collected episodes' mean returns reach the threshold,
#     # or every 1000 steps, we test it on test_collector
#     if collect_result['rews'].mean() >= env.spec.reward_threshold or i % 1000 == 0:
#         policy.set_eps(0.05)
#         result = test_collector.collect(n_episode=100)
#         if result['rews'].mean() >= env.spec.reward_threshold:
#             print(f'Finished training! Test mean returns: {result["rews"].mean()}')
#             break
#         else:
#             # back to training eps
#             policy.set_eps(0.1)
#
#     # train policy with a sampled batch data from buffer
#     losses = policy.update(64, train_collector.buffer)
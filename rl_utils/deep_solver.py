#from rl_utils.solver import *

from .solver import *

import tianshou as ts
from tianshou.data import Batch, ReplayBuffer
import torch
from torch import nn

class example_Net2(nn.Module):
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
        # idx= np.abs(obs.numpy())
        # idx=np.argwhere(idx==1)
        # for id in idx:
        #     print(id[-1])
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class example_Net1(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 32), nn.ReLU(inplace=True),
            nn.Linear(32, 32), nn.ReLU(inplace=True),
            nn.Linear(32, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        #print(obs)
        # idx= np.abs(obs.numpy())
        # idx=np.argwhere(idx==1)
        # for id in idx:
        #     print(id[-1])
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class deep_solver(Solver):
    def __init__(self, bandit_n, bidding_range=100, state_range=100,eps=0.1,
                 start_point=0,overbid=False, init_proba=0.0,
                 step_floor=10000, signal=False,
                 cumulative_round=0,lr=1e-3,discount_factor=0,model_name='DQN',update_frequent=200,device='cpu'
                 ):
        """
        build the deep learn based solver
        using tianshou as an example
        """

        # Bandit_n refers to the single players

        super(deep_solver, self).__init__(bandit_n)

        self.model_name=model_name

        assert 0. <= eps <= 1.0
        self.eps = eps
        self.t = 0
        self.device=device

        self.start_point = start_point  # Number of exploration steps


        self.lr=lr
        self.discount_factor=discount_factor
        self.update_frequent=update_frequent
        
        

        self.bidding_range = bidding_range #action space
        self.state_range=state_range  # obs state range after encode into 1 number

        self.cumulative_round = cumulative_round

        self.overbid = overbid
        self.signal = signal

        self.true_value_list = []

        self.decay = 0.5  # 3 * self.bandit_n #ori: 10
        self.step_floor = step_floor  # Number of steps to eval the avg. maybe eval_step_size
        self.estimates = [init_proba] * self.bandit_n  # Optimistic initialization
        # list len = n

        self.init_deep_policy(lr=self.lr)
        self.init_data_buffer()

    def init_deep_policy(self,lr=1e-3):
        self.net = example_Net1(state_shape=self.state_range, action_shape=self.bidding_range).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr) #optimizier

        if self.cumulative_round==0:
            #no cumulative round
            discount_factor=0
        else:
            discount_factor=self.discount_factor

        if self.model_name =='DQN':
            self.policy = ts.policy.DQNPolicy(self.net, self.optim, discount_factor=discount_factor, estimation_step=self.cumulative_round+1,
                                         target_update_freq=self.cumulative_round+2)  # no discount factor
            
            #update target network on 2 epoch 

            self.policy.set_eps(self.eps)

    def init_data_buffer(self):

        self.buf = ReplayBuffer(size=self.update_frequent)


    @property
    def estimated_probas(self):
        return self.estimates

    def one_hot_emb(self,obs,max_range=100):

        #print(obs)

        res = torch.zeros(1, max_range).to(self.device)

        res[0][obs] = 1


        return res


    def step_decay_eps(self,started_eps=0.5):
        # whether to adopt step decayed eps or fixed ones
        self.eps = started_eps **(self.t /  self.step_floor)


    def generate_action(self,obs=None,test=False,upper_bound=None,info={},terminated=0,truncated=0):
        # Obs refers to the true value

        #obs =state

        lower_bid = obs * self.bidding_range
        # Lower bid does not refer to the actual lower bid, but the lower bid on the
        # encoded space.
        if upper_bound is not None:
            upper_bound=int(upper_bound)

        elif self.overbid or self.signal:

            upper_bound = self.bidding_range +1
        else:
            upper_bound = obs +1 #

        # encoded into one hot embbdedding vector


        batch = Batch(obs=self.one_hot_emb(obs=obs,max_range=self.state_range), info=info, terminated=terminated, truncated=truncated)  # the first dimension is batch-size
        if test:
            self.policy.set_eps(eps=0)

        results = self.policy(batch)
        act = results.act[0]# policy.forward return a batch, use ".act" to extract the action

        estimation = results.logits[0]

        #record estimation
        self.estimates[lower_bid:lower_bid+self.bidding_range] = estimation.cpu().tolist()

        #deal with over bid:
        if self.overbid is False and self.signal is False:
            # not allow over bid and obs is the true value
            act = estimation[0:upper_bound].argmax().cpu()

        # record
        act=int(act)

        if not test:

            self.record_action(act)
        else:
            self.policy.set_eps(eps=self.eps) #change back

        return act

    def update_policy(self, i,reward,done=0,terminated=0,truncated=0,info={},obs_next=0,immediate=0):  # do  record_action(self,i) before update policy

        action= i % self.bidding_range
        state = int(i / self.bidding_range)


        self.t+=1
        #record results into buffer
        data = Batch(obs=self.one_hot_emb(obs=state,max_range=self.state_range),
                     act=action, rew=reward, done=done,
                     obs_next=self.one_hot_emb(obs=state,max_range=self.state_range), info=info,
              terminated=terminated, truncated=truncated)

        self.buf.add(data) #automatic push the old data when the buff list is fullfilled

        self.step_decay_eps()


        if self.t % self.update_frequent==0 or immediate==1:
            # update batch
            losses = self.policy.update(sample_size=self.update_frequent, buffer=self.buf)
            if self.t % self.step_floor ==0:
                print('cuurent step is '+str(self.t) +' loss is '+str(losses))
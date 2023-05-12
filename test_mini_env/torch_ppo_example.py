import torch
from torch import nn

import random
import os
import numpy as np

import os
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from copy import deepcopy


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr, fc1_dims=128, fc2_dims=128, device='cpu'):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, fc1_dims=128, fc2_dims=128, device='cpu'):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state):
        value = self.critic(state)
        return value


class example_Net(nn.Module):
    def __init__(self, input_range, output_range, device='cpu', lr=1e-4, batch_size=16):
        super().__init__()

        self.input_dim = input_range
        self.output_dim = output_range
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(np.prod(input_range), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(output_range)),
        )
        self.actor = ActorNetwork(n_actions=np.prod(output_range), input_dims=np.prod(input_range), lr=lr,
                                  device=device)
        self.critic = CriticNetwork(input_dims=np.prod(input_range), lr=lr, device=device)

    def obs_encode(self, obs):
        # input N*1 dim obs denotes as batch -> N*M tensor
        # eg.one-hot encode

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs)

        one_hot_emb = torch.nn.functional.one_hot(obs, num_classes=self.input_dim)  # [0, N-1]

        return one_hot_emb.float().to(self.device)

    def set_seed(args, seed=0):
        # Set seed for result reproducibility
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, obs, state=None, info={}):
        # #print(obs)
        state = self.obs_encode(obs)

        dist = self.actor(state)
        value = self.critic(state).squeeze(-1)
        action = dist.sample()

        logits = dist.log_prob(action)
        return logits, value, action, dist


class pytorch_algorithm_class(object):
    def __init__(self, args, max_bidding_value, seed):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = example_Net(input_range=args.public_signal_range, output_range=max_bidding_value, device=device,
                                 lr=args.lr, batch_size=16)
        self.model.set_seed(seed=seed)  # if set seed

        self.loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss() #nn.MSELoss()

        self.loss_fn2 = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss() #nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.model.to(device)

        self.max_bidding_value = max_bidding_value


def get_custom_agent_name():
    return 'player_2'


def receive_observation_pytorch(self, args, budget, agent_name, agt_idx,
                                extra_info, observation, public_signal):
    obs = dict()
    obs['observation'] = observation['observation']
    if budget is not None:
        budget.generate_budeget(user_id=agent_name)
        obs['budget'] = budget.get_budget(user_id=agent_name)  # observe the new budget

    if args.communication == 1 and agt_idx in args.cm_id and extra_info is not None:
        obs['extra_info'] = extra_info[agent_name]['others_value']

    if args.public_signal:
        obs['public_signal'] = public_signal

    self.record_obs(obs)

    return obs


def get_custom_algorithm_in_pytorch(args, algorithm='DQN', seed=0):
    # build a algorithm for the rllib

    return pytorch_algorithm_class(args, max_bidding_value=args.bidding_range, seed=seed)


def generate_action_pytorch(self, obs):
    # print('self', self)
    # print('obs', obs)
    # encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

    signal = obs['public_signal']

    # action, _, _ = self.algorithm.compute_single_action(
    #     encoded_input_emb(signal, action_space_num=self.args.bidding_range)
    # )

    state = torch.tensor(obs['public_signal']).cuda()
    self.algorithm.model.eval()

    if self.record_data_num < self.args.exploration_epoch:

        action = random.randint(0, self.args.bidding_range - 1)

        # repeat until action equalling to the action

        with torch.no_grad():

            # tensor_action = self.algorithm.model(signal)[2]  # return as tensor without grad

            _, max_value, _, dist = self.algorithm.model(state)

            tensor_action = torch.tensor(action).cuda()
            #
            logsit = dist.log_prob(tensor_action)

           # print(logsit)






    else:
        # print('start !')

        with torch.no_grad():

            #tensor_action = self.algorithm.model(signal)[2]  # return as tensor without grad


            logsit, max_value, tensor_action, dist = self.algorithm.model(state)

    state = state.cpu().detach().numpy()
    logsit = logsit.cpu().detach().numpy()
    max_value = max_value.cpu().detach().numpy()
    record_action = tensor_action.cpu().detach().numpy()

    self.record_signal_ppo(signal=state, action=record_action, prob=logsit, value=max_value, reward=None, dist=dist)

    action = int(tensor_action.item())

    # Decode the encoded action
    # action = encoded_action % (self.algorithm.bidding_range)

    self.record_action(action)

    return action


def update_policy_pytorch(self, obs, reward, done, policy_clip=0.2):
    if obs['allocation'] == 1 or ('true_value' in obs) :  # skip the first round or knows the true value
        # only record allocation =1 or during exploration epoch

        self.record_signal(obs['public_signal'])
        if 'final_pay' in obs:
            payment = obs['final_pay']

            self.record_payment(abs(payment))  # abs

        ### try to optimize best convegence
        if 'true_value' in obs:
            # knows true value
            self.record_true_value(obs['true_value'])

            if obs['allocation'] == 0:
                # and not allocate
                action = self.action_history[-1]

                #  pay =5 bid=4 v=6
                #
                #reward = min(0,obs['true_value'] - action) #punish the action that over bid | if bid correct
                    #obs['true_value'] - action  # assume if win

        self.record_reward(reward)

        # with torch.enable_grad():
        #     state = torch.tensor(obs['public_signal']).cuda()
        #     logsit, max_value, action, dist = self.algorithm.model(state)
        #
        #     state = state.cpu().detach().numpy()
        #     logsit = logsit.cpu().detach().numpy()
        #     max_value = max_value.cpu().detach().numpy()
        #     action = action.cpu().detach().numpy()

        self.record_signal_ppo(signal=None, action=None, prob=None, value=None, reward=reward,dist=None)

        self.record_data_num += 1

        batchsize = self.args.batchsize
        new_loss = self.args.cfr_loss

        old = 0  # use wangxue's implement

        if self.record_data_num % batchsize == 0:  # self.algrithm.update_frequency:

            self.algorithm.model.train()
            # self.algorithm.model.cuda()
            with torch.enable_grad():

                if old:
                    # get results from the whole action history
                    idx = torch.randperm(len(self.signal_history_ppo), device='cuda')[
                          :batchsize].cpu().numpy()  # np.random.permutation(len(self.signal_history_ppo))[:batchsize]

                    input_batch = torch.tensor(self.signal_history_ppo[idx], dtype=torch.int64).to('cuda')
                    # corresponding action
                    action_batch = torch.tensor(self.action_history_ppo[idx]).to('cuda')
                    old_prob_batch = torch.tensor(self.prob_history_ppo[idx]).to('cuda')
                    # corresponding computed value
                    vals_batch = torch.tensor(self.value_history_ppo[idx]).to('cuda')
                    reward_batch = torch.tensor(self.reward_histoy_ppo[idx]).to('cuda')
                else:

                    # eric modify version
                    buffer_len = min(len(self.signal_history_ppo), 20000)

                    # idx = np.random.permutation(buffer_len)[:batchsize]
                    idx = torch.randperm(buffer_len, device='cuda')[:batchsize].cpu().numpy()

                    tmp_signal_history = self.signal_history_ppo[-buffer_len:]
                    tmp_prob_history = self.prob_history_ppo[-buffer_len:]
                    tmp_action_history = self.action_history_ppo[-buffer_len:]
                    tmp_value_history = self.value_history_ppo[-buffer_len:]
                    tmp_reward_history = self.reward_histoy_ppo[-buffer_len:]

                    old_dist_history = self.dist_history_ppo[-buffer_len:]

                    pay_history = self.payment_history[-buffer_len:]
                    pay_his = np.array(pay_history)

                    pay_batch =torch.tensor(pay_his[idx]).to('cuda')




                    true_value_history = deepcopy(self.true_value_list[-buffer_len:])
                    true_value_history = np.array(true_value_history)
                    # print(true_value_history)
                    # print(self.true_value_list)

                    true_value_batch = torch.tensor(true_value_history[idx]).to('cuda')

                    input_batch = torch.tensor(tmp_signal_history[idx], dtype=torch.int64).to('cuda')
                    # corresponding action
                    action_batch = torch.tensor(tmp_action_history[idx]).to('cuda')
                    old_prob_batch = torch.tensor(tmp_prob_history[idx]).to('cuda')
                    # corresponding computed value
                    vals_batch = torch.tensor(tmp_value_history[idx]).to('cuda')
                    reward_batch = torch.tensor(tmp_reward_history[idx]).to('cuda')

                    if new_loss:
                        ## cfr config
                        K = 1
                        second_flag = False
                        if self.args.mechanism == 'second_price':
                           second_flag = True
                        #
                        logsit_size = self.args.bidding_range

                        pay = pay_batch #true_value_batch - reward_batch  # reward = true_value - pay & may virtual as we assume user win thus the payment is higher than the computed pay

                        virtual_bid = torch.tensor([j for j in range(logsit_size)]).reshape(1, K, -1).cuda()
                        win_bid_idx = virtual_bid - pay.unsqueeze(-1)

                        if second_flag:
                            val_diff = (true_value_batch - pay).unsqueeze(-1).repeat(1, 1, win_bid_idx.shape[-1])
                            # tmp = true_value.unsqueeze(-1).repeat(1, 1, win_bid_idx.shape[-1]) - virtual_bid
                        else:
                            val_diff = true_value_batch.unsqueeze(-1).repeat(1, 1, win_bid_idx.shape[-1]) - virtual_bid
                        val_diff = val_diff.float()
                        idea_logsit1 = torch.zeros(val_diff.size()).cuda()
                        idea_logsit1[win_bid_idx >= 0] = val_diff[win_bid_idx >= 0]


                # compute the new distrbution

                logsit, max_value, pred, dist = self.algorithm.model(input_batch.to('cuda'))

                new_probs = dist.log_prob(action_batch)
                prob_ratio = new_probs.exp() / old_prob_batch.exp() # 作出这个行动的前后概率变化

                if (not old) and new_loss:
                    # try cfr loss v1
                    sft = nn.Softmax(dim=-1)
                    cfr_prob_list = sft(idea_logsit1)  # K,batchsize, bidding_range

                    # deep copy for tmp
                    tmp_action_batch = deepcopy(action_batch)
                    tmp_action_batch = tmp_action_batch.int().cpu()

                    # new distribution whole
                    current_distrbution = deepcopy(dist.logits.detach().exp()) #[batchsize,bidding_range]
                    #current_distrbution=current_distrbution.detach()

                    #print(dist.logits.grad)
                    #print(1/0)

                    #print(current_distrbution.size())


                    cfr_reward_batch = torch.zeros(reward_batch.size()).cuda()

                    cfr_prob_batch = old_prob_batch.exp()
                    for kk in range(batchsize):
                        cfr_prob_batch[kk] = cfr_prob_list[0, kk, int(tmp_action_batch[kk])]  # update cfr prob

                        #cfr_reward_batch[kk] =max(idea_logsit1[0][kk])  #version1:  the ideally max reward rather than received reward

                        # version 2: possibility sum + apply real results
                        idea_logsit1[0, kk, int(tmp_action_batch[kk])] = reward_batch[kk]  # apply real results

                        #cfr_reward_batch[kk] = torch.sum(idea_logsit1[0][kk] * old_prob_batch[kk])  # use old possibility * cfr_reward
                        cfr_reward_batch[kk] = torch.sum(
                            idea_logsit1[0][kk] * current_distrbution[kk])  # use new possibility * cfr_reward



                    cfr_prob_ratio = new_probs.exp() / cfr_prob_batch

                    #### cfr loss v2


                    # first compute the
                    #
                    #
                    # v1
                    # advantage = cfr_reward_batch - vals_batch
                    #
                    # weighted_probs = advantage * cfr_prob_ratio
                    #
                    # weighted_clipped_probs = torch.clamp(cfr_prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage

                    # v2
                    advantage = reward_batch - vals_batch

                    weighted_probs = advantage * prob_ratio

                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage


                    # v2
                    # old_cfr_reward_batch = torch.zeros(reward_batch.size()).cuda()
                    # new_cfr_reward_batch = torch.zeros(reward_batch.size()).cuda()
                    # for kk in range(batchsize):
                    #     old_cfr_prob = old_dist_history[idx[kk]].logits.exp()
                    #     new_cfr_prob = (dist.logits.exp()) #[batchsize,bidding_range]
                    #
                    #     tmp1 = idea_logsit1[0][kk]
                    #
                    #
                    #
                    #     #old_cfr_reward_batch[kk] = torch.sum(idea_logsit1[0][kk] *old_cfr_prob )
                    #     new_cfr_reward_batch[kk] = torch.sum(idea_logsit1[0][kk] *new_cfr_prob )
                    #
                    # #
                    # advantage = reward_batch - vals_batch
                    #
                    # weighted_probs = advantage * prob_ratio
                    #
                    # weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage


                    actor_loss = torch.mean(-torch.min(weighted_probs, weighted_clipped_probs))
                    critic_loss = torch.mean(
                        (cfr_reward_batch - max_value) ** 2)  # + torch.mean((reward_batch - max_value) ** 2)

                else:
                    advantage = reward_batch - vals_batch

                    weighted_probs = advantage * prob_ratio

                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage

                    actor_loss = torch.mean(-torch.min(weighted_probs, weighted_clipped_probs))
                    critic_loss = torch.mean((reward_batch - max_value) ** 2)

                total_loss = actor_loss + 0.5 * critic_loss
                self.algorithm.model.actor.optimizer.zero_grad()
                self.algorithm.model.critic.optimizer.zero_grad()
                total_loss.backward()

                self.algorithm.model.actor.optimizer.step()
                self.algorithm.model.critic.optimizer.step()

                if self.record_data_num % (batchsize * 100) == 0:
                    print(
                        f'agent({self.agent_name}) epoch {self.record_data_num / batchsize} --> the training loss(mse)  with batch size {batchsize} is {total_loss.item()}')

    return


def update_policy_pytorch_v2(self, obs, reward, done, policy_clip=0.2):
    if obs['allocation'] == 1 or ('true_value' in obs) :  # skip the first round or knows the true value
        # only record allocation =1 or during exploration epoch

        self.record_signal(obs['public_signal'])
        if 'final_pay' in obs:
            payment = obs['final_pay']

            self.record_payment(abs(payment))  # abs

        ### try to optimize best convegence
        if 'true_value' in obs:
            # knows true value
            self.record_true_value(obs['true_value'])

            if obs['allocation'] == 0:
                # and not allocate
                action = self.action_history[-1]


        self.record_reward(reward)



        self.record_signal_ppo(signal=None, action=None, prob=None, value=None, reward=reward,dist=None)

        self.record_data_num += 1

        batchsize = self.args.batchsize
        new_loss = self.args.cfr_loss

        old = 0  # use wangxue's implement
        buffer_size =10000
        epochs=400


        if self.record_data_num % buffer_size == 0:  # self.algrithm.update_frequency:

           self.algorithm.model.train()
            # self.algorithm.model.cuda()
           for eps in range(epochs):
               with torch.enable_grad():

                   if old:
                       # get results from the whole action history
                       idx = torch.randperm(len(self.signal_history_ppo), device='cuda')[
                             :batchsize].cpu().numpy()  # np.random.permutation(len(self.signal_history_ppo))[:batchsize]

                       input_batch = torch.tensor(self.signal_history_ppo[idx], dtype=torch.int64).to('cuda')
                       # corresponding action
                       action_batch = torch.tensor(self.action_history_ppo[idx]).to('cuda')
                       old_prob_batch = torch.tensor(self.prob_history_ppo[idx]).to('cuda')
                       # corresponding computed value
                       vals_batch = torch.tensor(self.value_history_ppo[idx]).to('cuda')
                       reward_batch = torch.tensor(self.reward_histoy_ppo[idx]).to('cuda')
                   else:

                       # eric modify version
                       buffer_len = min(len(self.signal_history_ppo), buffer_size)

                       # idx = np.random.permutation(buffer_len)[:batchsize]
                       idx = torch.randperm(buffer_len, device='cuda')[:batchsize].cpu().numpy()

                       tmp_signal_history = self.signal_history_ppo[-buffer_len:]
                       tmp_prob_history = self.prob_history_ppo[-buffer_len:]
                       tmp_action_history = self.action_history_ppo[-buffer_len:]
                       tmp_value_history = self.value_history_ppo[-buffer_len:]
                       tmp_reward_history = self.reward_histoy_ppo[-buffer_len:]

                       old_dist_history = self.dist_history_ppo[-buffer_len:]

                       pay_history = self.payment_history[-buffer_len:]
                       pay_his = np.array(pay_history)

                       pay_batch = torch.tensor(pay_his[idx]).to('cuda')

                       true_value_history = deepcopy(self.true_value_list[-buffer_len:])
                       true_value_history = np.array(true_value_history)
                       # print(true_value_history)
                       # print(self.true_value_list)

                       true_value_batch = torch.tensor(true_value_history[idx]).to('cuda')

                       input_batch = torch.tensor(tmp_signal_history[idx], dtype=torch.int64).to('cuda')
                       # corresponding action
                       action_batch = torch.tensor(tmp_action_history[idx]).to('cuda')
                       old_prob_batch = torch.tensor(tmp_prob_history[idx]).to('cuda')
                       # corresponding computed value
                       vals_batch = torch.tensor(tmp_value_history[idx]).to('cuda')
                       reward_batch = torch.tensor(tmp_reward_history[idx]).to('cuda')

                       if new_loss:
                           ## cfr config
                           K = 1
                           second_flag = False
                           if self.args.mechanism == 'second_price':
                               second_flag = True
                           #
                           logsit_size = self.args.bidding_range

                           pay = pay_batch  # true_value_batch - reward_batch  # reward = true_value - pay & may virtual as we assume user win thus the payment is higher than the computed pay

                           virtual_bid = torch.tensor([j for j in range(logsit_size)]).reshape(1, K, -1).cuda()
                           win_bid_idx = virtual_bid - pay.unsqueeze(-1)

                           if second_flag:
                               val_diff = (true_value_batch - pay).unsqueeze(-1).repeat(1, 1, win_bid_idx.shape[-1])
                               # tmp = true_value.unsqueeze(-1).repeat(1, 1, win_bid_idx.shape[-1]) - virtual_bid
                           else:
                               val_diff = true_value_batch.unsqueeze(-1).repeat(1, 1,
                                                                                win_bid_idx.shape[-1]) - virtual_bid
                           val_diff = val_diff.float()
                           idea_logsit1 = torch.zeros(val_diff.size()).cuda()
                           idea_logsit1[win_bid_idx >= 0] = val_diff[win_bid_idx >= 0]

                   # compute the new distrbution

                   logsit, max_value, pred, dist = self.algorithm.model(input_batch.to('cuda'))

                   new_probs = dist.log_prob(action_batch)
                   prob_ratio = new_probs.exp() / old_prob_batch.exp()  # 作出这个行动的前后概率变化

                   if (not old) and new_loss:
                       # try cfr loss v1
                       sft = nn.Softmax(dim=-1)
                       cfr_prob_list = sft(idea_logsit1)  # K,batchsize, bidding_range

                       # deep copy for tmp
                       tmp_action_batch = deepcopy(action_batch)
                       tmp_action_batch = tmp_action_batch.int().cpu()

                       # new distribution whole
                       current_distrbution = deepcopy(dist.logits.detach().exp())  # [batchsize,bidding_range]
                       # current_distrbution=current_distrbution.detach()

                       # print(dist.logits.grad)
                       # print(1/0)

                       # print(current_distrbution.size())

                       cfr_reward_batch = torch.zeros(reward_batch.size()).cuda()

                       cfr_prob_batch = old_prob_batch.exp()
                       for kk in range(batchsize):
                           cfr_prob_batch[kk] = cfr_prob_list[0, kk, int(tmp_action_batch[kk])]  # update cfr prob

                           # cfr_reward_batch[kk] =max(idea_logsit1[0][kk])  #version1:  the ideally max reward rather than received reward

                           # version 2: possibility sum + apply real results
                           idea_logsit1[0, kk, int(tmp_action_batch[kk])] = reward_batch[kk]  # apply real results

                           # cfr_reward_batch[kk] = torch.sum(idea_logsit1[0][kk] * old_prob_batch[kk])  # use old possibility * cfr_reward
                           cfr_reward_batch[kk] = torch.sum(
                               idea_logsit1[0][kk] * current_distrbution[kk])  # use new possibility * cfr_reward

                       cfr_prob_ratio = new_probs.exp() / cfr_prob_batch

                       #### cfr loss v2

                       # first compute the
                       #
                       #
                       # v1
                       # advantage = cfr_reward_batch - vals_batch
                       #
                       # weighted_probs = advantage * cfr_prob_ratio
                       #
                       # weighted_clipped_probs = torch.clamp(cfr_prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage

                       # v2
                       advantage = reward_batch - vals_batch

                       weighted_probs = advantage * prob_ratio

                       weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage

                       # v2
                       # old_cfr_reward_batch = torch.zeros(reward_batch.size()).cuda()
                       # new_cfr_reward_batch = torch.zeros(reward_batch.size()).cuda()
                       # for kk in range(batchsize):
                       #     old_cfr_prob = old_dist_history[idx[kk]].logits.exp()
                       #     new_cfr_prob = (dist.logits.exp()) #[batchsize,bidding_range]
                       #
                       #     tmp1 = idea_logsit1[0][kk]
                       #
                       #
                       #
                       #     #old_cfr_reward_batch[kk] = torch.sum(idea_logsit1[0][kk] *old_cfr_prob )
                       #     new_cfr_reward_batch[kk] = torch.sum(idea_logsit1[0][kk] *new_cfr_prob )
                       #
                       # #
                       # advantage = reward_batch - vals_batch
                       #
                       # weighted_probs = advantage * prob_ratio
                       #
                       # weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage

                       actor_loss = torch.mean(-torch.min(weighted_probs, weighted_clipped_probs))
                       critic_loss = torch.mean(
                           (cfr_reward_batch - max_value) ** 2)  # + torch.mean((reward_batch - max_value) ** 2)

                   else:
                       advantage = reward_batch - vals_batch

                       weighted_probs = advantage * prob_ratio

                       weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage

                       actor_loss = torch.mean(-torch.min(weighted_probs, weighted_clipped_probs))
                       critic_loss = torch.mean((reward_batch - max_value) ** 2)

                   total_loss = actor_loss + 0.5 * critic_loss
                   self.algorithm.model.actor.optimizer.zero_grad()
                   self.algorithm.model.critic.optimizer.zero_grad()
                   total_loss.backward()

                   self.algorithm.model.actor.optimizer.step()
                   self.algorithm.model.critic.optimizer.step()

                   if (eps+1) %  100 == 0:
                       print(
                           f'agent({self.agent_name}) in epoch {self.record_data_num / buffer_size} of step {eps+1} --> the training loss(mse)  with batch size {batchsize} is {total_loss.item()}')

    return

def update_policy_pytorch_v_diff(self, obs, reward, done, policy_clip=0.2):
    if obs['allocation'] == 1 or ('true_value' in obs) :  # skip the first round or knows the true value
        # only record allocation =1 or during exploration epoch

        self.record_signal(obs['public_signal'])
        if 'final_pay' in obs:
            payment = obs['final_pay']

            self.record_payment(abs(payment))  # abs

        ### try to optimize best convegence
        if 'true_value' in obs:
            # knows true value
            self.record_true_value(obs['true_value'])

            if obs['allocation'] == 0:
                # and not allocate
                action = self.action_history[-1]


        self.record_reward(reward)



        self.record_signal_ppo(signal=None, action=None, prob=None, value=None, reward=reward,dist=None)

        self.record_data_num += 1

        batchsize = self.args.batchsize
        new_loss = self.args.cfr_loss

        old = 0  # use wangxue's implement
        # diff
        if int(self.agent_name[-1])==0:
            buffer_size =1000
        elif int(self.agent_name[-1])==3:
            buffer_size=20000
        else:

            buffer_size =int(self.agent_name[-1]) * 5000

        epochs=400


        if self.record_data_num % buffer_size == 0:  # self.algrithm.update_frequency:

           self.algorithm.model.train()
            # self.algorithm.model.cuda()
           for eps in range(epochs):
               with torch.enable_grad():

                   if old:
                       # get results from the whole action history
                       idx = torch.randperm(len(self.signal_history_ppo), device='cuda')[
                             :batchsize].cpu().numpy()  # np.random.permutation(len(self.signal_history_ppo))[:batchsize]

                       input_batch = torch.tensor(self.signal_history_ppo[idx], dtype=torch.int64).to('cuda')
                       # corresponding action
                       action_batch = torch.tensor(self.action_history_ppo[idx]).to('cuda')
                       old_prob_batch = torch.tensor(self.prob_history_ppo[idx]).to('cuda')
                       # corresponding computed value
                       vals_batch = torch.tensor(self.value_history_ppo[idx]).to('cuda')
                       reward_batch = torch.tensor(self.reward_histoy_ppo[idx]).to('cuda')
                   else:

                       # eric modify version
                       buffer_len = min(len(self.signal_history_ppo), buffer_size)

                       # idx = np.random.permutation(buffer_len)[:batchsize]
                       idx = torch.randperm(buffer_len, device='cuda')[:batchsize].cpu().numpy()

                       tmp_signal_history = self.signal_history_ppo[-buffer_len:]
                       tmp_prob_history = self.prob_history_ppo[-buffer_len:]
                       tmp_action_history = self.action_history_ppo[-buffer_len:]
                       tmp_value_history = self.value_history_ppo[-buffer_len:]
                       tmp_reward_history = self.reward_histoy_ppo[-buffer_len:]

                       old_dist_history = self.dist_history_ppo[-buffer_len:]

                       pay_history = self.payment_history[-buffer_len:]
                       pay_his = np.array(pay_history)

                       pay_batch = torch.tensor(pay_his[idx]).to('cuda')

                       true_value_history = deepcopy(self.true_value_list[-buffer_len:])
                       true_value_history = np.array(true_value_history)
                       # print(true_value_history)
                       # print(self.true_value_list)

                       true_value_batch = torch.tensor(true_value_history[idx]).to('cuda')

                       input_batch = torch.tensor(tmp_signal_history[idx], dtype=torch.int64).to('cuda')
                       # corresponding action
                       action_batch = torch.tensor(tmp_action_history[idx]).to('cuda')
                       old_prob_batch = torch.tensor(tmp_prob_history[idx]).to('cuda')
                       # corresponding computed value
                       vals_batch = torch.tensor(tmp_value_history[idx]).to('cuda')
                       reward_batch = torch.tensor(tmp_reward_history[idx]).to('cuda')

                       if new_loss:
                           ## cfr config
                           K = 1
                           second_flag = False
                           if self.args.mechanism == 'second_price':
                               second_flag = True
                           #
                           logsit_size = self.args.bidding_range

                           pay = pay_batch  # true_value_batch - reward_batch  # reward = true_value - pay & may virtual as we assume user win thus the payment is higher than the computed pay

                           virtual_bid = torch.tensor([j for j in range(logsit_size)]).reshape(1, K, -1).cuda()
                           win_bid_idx = virtual_bid - pay.unsqueeze(-1)

                           if second_flag:
                               val_diff = (true_value_batch - pay).unsqueeze(-1).repeat(1, 1, win_bid_idx.shape[-1])
                               # tmp = true_value.unsqueeze(-1).repeat(1, 1, win_bid_idx.shape[-1]) - virtual_bid
                           else:
                               val_diff = true_value_batch.unsqueeze(-1).repeat(1, 1,
                                                                                win_bid_idx.shape[-1]) - virtual_bid
                           val_diff = val_diff.float()
                           idea_logsit1 = torch.zeros(val_diff.size()).cuda()
                           idea_logsit1[win_bid_idx >= 0] = val_diff[win_bid_idx >= 0]

                   # compute the new distrbution

                   logsit, max_value, pred, dist = self.algorithm.model(input_batch.to('cuda'))

                   new_probs = dist.log_prob(action_batch)
                   prob_ratio = new_probs.exp() / old_prob_batch.exp()  # 作出这个行动的前后概率变化

                   if (not old) and new_loss:
                       # try cfr loss v1
                       sft = nn.Softmax(dim=-1)
                       cfr_prob_list = sft(idea_logsit1)  # K,batchsize, bidding_range

                       # deep copy for tmp
                       tmp_action_batch = deepcopy(action_batch)
                       tmp_action_batch = tmp_action_batch.int().cpu()

                       # new distribution whole
                       current_distrbution = deepcopy(dist.logits.detach().exp())  # [batchsize,bidding_range]
                       # current_distrbution=current_distrbution.detach()

                       # print(dist.logits.grad)
                       # print(1/0)

                       # print(current_distrbution.size())

                       cfr_reward_batch = torch.zeros(reward_batch.size()).cuda()

                       cfr_prob_batch = old_prob_batch.exp()
                       for kk in range(batchsize):
                           cfr_prob_batch[kk] = cfr_prob_list[0, kk, int(tmp_action_batch[kk])]  # update cfr prob

                           # cfr_reward_batch[kk] =max(idea_logsit1[0][kk])  #version1:  the ideally max reward rather than received reward

                           # version 2: possibility sum + apply real results
                           idea_logsit1[0, kk, int(tmp_action_batch[kk])] = reward_batch[kk]  # apply real results

                           # cfr_reward_batch[kk] = torch.sum(idea_logsit1[0][kk] * old_prob_batch[kk])  # use old possibility * cfr_reward
                           cfr_reward_batch[kk] = torch.sum(
                               idea_logsit1[0][kk] * current_distrbution[kk])  # use new possibility * cfr_reward

                       cfr_prob_ratio = new_probs.exp() / cfr_prob_batch

                       #### cfr loss v2

                       # first compute the
                       #
                       #
                       # v1
                       # advantage = cfr_reward_batch - vals_batch
                       #
                       # weighted_probs = advantage * cfr_prob_ratio
                       #
                       # weighted_clipped_probs = torch.clamp(cfr_prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage

                       # v2
                       advantage = reward_batch - vals_batch

                       weighted_probs = advantage * prob_ratio

                       weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage

                       # v2
                       # old_cfr_reward_batch = torch.zeros(reward_batch.size()).cuda()
                       # new_cfr_reward_batch = torch.zeros(reward_batch.size()).cuda()
                       # for kk in range(batchsize):
                       #     old_cfr_prob = old_dist_history[idx[kk]].logits.exp()
                       #     new_cfr_prob = (dist.logits.exp()) #[batchsize,bidding_range]
                       #
                       #     tmp1 = idea_logsit1[0][kk]
                       #
                       #
                       #
                       #     #old_cfr_reward_batch[kk] = torch.sum(idea_logsit1[0][kk] *old_cfr_prob )
                       #     new_cfr_reward_batch[kk] = torch.sum(idea_logsit1[0][kk] *new_cfr_prob )
                       #
                       # #
                       # advantage = reward_batch - vals_batch
                       #
                       # weighted_probs = advantage * prob_ratio
                       #
                       # weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage

                       actor_loss = torch.mean(-torch.min(weighted_probs, weighted_clipped_probs))
                       critic_loss = torch.mean(
                           (cfr_reward_batch - max_value) ** 2)  # + torch.mean((reward_batch - max_value) ** 2)

                   else:
                       advantage = reward_batch - vals_batch

                       weighted_probs = advantage * prob_ratio

                       weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage

                       actor_loss = torch.mean(-torch.min(weighted_probs, weighted_clipped_probs))
                       critic_loss = torch.mean((reward_batch - max_value) ** 2)

                   total_loss = actor_loss + 0.5 * critic_loss
                   self.algorithm.model.actor.optimizer.zero_grad()
                   self.algorithm.model.critic.optimizer.zero_grad()
                   total_loss.backward()

                   self.algorithm.model.actor.optimizer.step()
                   self.algorithm.model.critic.optimizer.step()

                   if (eps+1) %  100 == 0:
                       print(
                           f'agent({self.agent_name}) in epoch {self.record_data_num / buffer_size} of step {eps+1} --> the training loss(mse)  with batch size {batchsize} is {total_loss.item()}')

    return


def test_policy_pytorch(self, state):
    # signal = state['public_signal']
    # in the current plot function, test input is the signal
    # will modified in the future plot module

    signal = state

    # action, _, _ = self.algo.compute_single_action(
    #     encoded_input_emb(signal, action_space_num=self.args.bidding_range)
    # )
    self.algorithm.model.eval()
    with torch.no_grad():
        tensor_action = self.algorithm.model(signal)[2]  # return as tensor without grad

    action = int(tensor_action.item())

    return action

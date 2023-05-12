import torch
from torch import nn

import random
import os
import numpy as np


class example_Net(nn.Module):
    def __init__(self, input_range, output_range,device='cpu'):
        super().__init__()

        self.input_dim=input_range
        self.output_dim=output_range
        self.device=device


        self.model = nn.Sequential(
            nn.Linear(np.prod(input_range), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(output_range)),
        )

    def obs_encode(self,obs):
        # input N*1 dim obs denotes as batch -> N*M tensor
        # eg.one-hot encode

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs)

        one_hot_emb = torch.nn.functional.one_hot(obs,num_classes=self.input_dim) #[0, N-1]

        return one_hot_emb.float().to(self.device)


    def set_seed(args,seed=0):
        # Set seed for result reproducibility
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



    def forward(self, obs, state=None, info={}):
        #print(obs)


        encoded_obs = self.obs_encode(obs)
        #batch = encoded_obs.shape[0]


        # compute q value
        logits = self.model(encoded_obs)

        # argmax q value
        max_value,act = logits.max(dim=-1)

        # batchsize *1

        return logits,max_value,act



class pytorch_algorithm_class(object):
        def __init__(self,args,max_bidding_value,seed):

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            self.model = example_Net( input_range=args.public_signal_range, output_range=max_bidding_value,device=device)
            self.model.set_seed(seed=seed) # if set seed

            self.loss_fn =nn.MSELoss()#nn.CrossEntropyLoss() #nn.MSELoss()

            self.loss_fn2 = nn.CrossEntropyLoss() # nn.CrossEntropyLoss() #nn.MSELoss()

            self.optimizer=torch.optim.Adam(self.model.parameters(), lr=args.lr)

            self.model.to(device)

            self.max_bidding_value=max_bidding_value

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

def get_custom_algorithm_in_pytorch(args,algorithm='DQN',seed=0):

    # build a algorithm for the rllib




    return pytorch_algorithm_class(args,max_bidding_value=args.bidding_range,seed=seed)

def generate_action_pytorch(self, obs):
    # print('self', self)
    # print('obs', obs)
    # encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

    signal = obs['public_signal']

    # action, _, _ = self.algorithm.compute_single_action(
    #     encoded_input_emb(signal, action_space_num=self.args.bidding_range)
    # )
    if self.record_data_num<self.args.exploration_epoch:

        action = random.randint(0,self.args.bidding_range-1)

    else:
        #print('start !')
        self.algorithm.model.eval()
        with torch.no_grad():
            logits, max_value, tensor_action = self.algorithm.model(signal)  # return as tensor without grad

        action = int(tensor_action.item())


    # Decode the encoded action
    #action = encoded_action % (self.algorithm.bidding_range)

    self.record_action(action)

    return action


def update_policy_pytorch(self, obs, reward, done):
    #last_action = self.get_latest_action()
    # true_value = []
    #
    # if done:
    #     true_value = self.get_latest_true_value()
    # else:
    #     true_value = self.get_last_round_true_value()
    # encoded_action = last_action + true_value * (self.algorithm.bidding_range)
    # self.algorithm.update_policy(encoded_action, reward=reward)

    #tt = 'true_value' in obs
    #bb = self.record_data_num < self.args.exploration_epoch

    if obs['allocation']==1 or ('true_value' in obs) : #skip the first round or knows the true value
        #only record allocation =1 or during exploration epoch


        self.record_signal(obs['public_signal'])


        ### try to optimize best convegence
        if 'true_value' in obs :
            #knows true value
            self.record_true_value(obs['true_value'])

            if obs['allocation']==0:
                #and not allocate
                action=self.action_history[-1]
                reward = obs['true_value']-action #assume if win

        self.record_reward(reward)

        self.record_data_num+=1

        batchsize=self.args.batchsize
        new_loss=self.args.cfr_loss

        #self.args.exploration_epoch> self.record_data_num  #try to use two type loss [loss1:reward no-regret loss | loss 2: reconstruct loss]

        if self.record_data_num % batchsize==0 : #self.algrithm.update_frequency:
            input_batch = self.signal_history[-batchsize:]
            y_batch = self.reward_history[-batchsize:]

            # rebuild into pytorch version

            if not isinstance(input_batch, torch.Tensor):
                input_batch=torch.tensor(input_batch)

            if not isinstance(y_batch, torch.Tensor):
                y_batch = torch.tensor(y_batch)

                #            #for better train
                true_value = torch.tensor( self.true_value_list[-batchsize:])+1 #start from 1
                true_value=true_value.float().cuda().reshape([batchsize, 1])


                #y_batch = y_batch / true_value #use utility/value rate


            y_batch=y_batch.float().cuda().reshape([batchsize,1]) #/ self.algorithm.max_bidding_value

            self.algorithm.model.train()
            self.algorithm.model.cuda()
            with torch.enable_grad():
                logsit,max_value,pred = self.algorithm.model(input_batch) # input with grad

            #expected_reward = logsit[pred]


           # loss = self.algorithm.loss_fn(max_value, y_batch) #compute  reward - expected reward loss




            # algorithm Backpropagation
            self.algorithm.optimizer.zero_grad() #optimizer init
            #loss.backward() #compute loss

            if (new_loss):
                #consider the reconstruct loss
                # pay = true_value - reward
                pay = true_value - y_batch #*true_value

                second_flag=False
                if self.args.mechanism=='second_price':
                    second_flag=True





                #idea_logsit = torch.zeros(logsit.size())

                #may optimize
                # for i in range(batchsize):
                #     for j in range(logsit.size()[-1]):
                #         if pay[i] > j: # pay > bid
                #             idea_logsit[i][0][j] = 0 # pay > bid = not win
                #         else:
                #             idea_logsit[i][0][j] = true_value[i] - pay[i] # can win more

                K = 1
                logsit_size = self.args.bidding_range

                virtual_bid = torch.tensor([j for j in range(logsit_size)]).reshape(1, K, -1).cuda()
                win_bid_idx = virtual_bid - pay.unsqueeze(-1)
                if second_flag:
                    val_diff = (true_value - pay).unsqueeze(-1).repeat(1, 1, win_bid_idx.shape[-1])
                    #tmp = true_value.unsqueeze(-1).repeat(1, 1, win_bid_idx.shape[-1]) - virtual_bid
                else:
                    val_diff = true_value.unsqueeze(-1).repeat(1, 1, win_bid_idx.shape[-1]) - virtual_bid

                # print(val_diff.shape)
                # print(tmp.shape)
                idea_logsit1 = torch.zeros(logsit.size()).cuda()
                idea_logsit1[win_bid_idx >= 0] = val_diff[win_bid_idx >= 0]
                #
                # for i in range(batchsize):
                #     for j in range(logsit.size()[-1]):
                #         if pay[i] <= j:
                #             if second_flag:
                #                 idea_logsit[i][0][j] = true_value[i] - pay[i]  # can win more
                #             else:
                #                 idea_logsit[i][0][j] = true_value[i] - j  # first price
                #
                # print(torch.mean(torch.abs(idea_logsit.cuda() - idea_logsit1)))
                # print(idea_logsit1.shape, idea_logsit.shape)

                #idea_logsit=idea_logsit.float().cuda()





                loss2=self.algorithm.loss_fn(logsit, idea_logsit1)
                loss2.backward()

                if self.record_data_num % (batchsize*1000) ==0:
                    print('agent(' + str(self.agent_name) + ') epoch ' + str(
                        self.record_data_num / batchsize) + ' --> the training loss(avg mse)  with batch size ' + str(
                        batchsize) + ' is ' + str(loss2.item()))
            else:
                loss = self.algorithm.loss_fn(max_value, y_batch)  # compute  reward - expected reward loss
                loss.backward()  # compute loss
                if self.record_data_num % (batchsize * 1000) == 0:
                    print('agent(' + str(self.agent_name) + ') epoch ' + str(
                        self.record_data_num / batchsize) + ' --> the training loss(mse)  with batch size ' + str(
                        batchsize) + ' is ' + str(loss.item()))

            self.algorithm.optimizer.step() #step






    # to be done:
        #update policy

    #


    return



def test_policy_pytorch(self,state):

    #signal = state['public_signal']
    # in the current plot function, test input is the signal
    # will modified in the future plot module

    signal =state


    # action, _, _ = self.algo.compute_single_action(
    #     encoded_input_emb(signal, action_space_num=self.args.bidding_range)
    # )
    self.algorithm.model.eval()
    with torch.no_grad():
        logits,max_value,tensor_action = self.algorithm.model(signal)  # return as tensor without grad

    action = int(tensor_action.item())


    return action
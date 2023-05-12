import torch
import os
import  random
import numpy as np
from torch import nn
from .deep_allocation import *

class example_payment_Net(nn.Module):
    def __init__(self,agent_number=4,device=torch.device('cpu')):
        super().__init__()

        self.agent_num=agent_number
        self.device=device

        self.model = nn.Sequential(
            nn.Linear(np.prod(self.agent_num), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(self.agent_num)),
            nn.Softmax()
        )

    def forward(self,state):
        # compute q value

        #get the softmax allocation possibility

        logits = self.model(state)

        # argmax q value
        #max_p ,winner = logits.max(dim=-1)

        # batchsize *1

        # return logits,max_p, winner
        return logits


class deep_payment(object):

    def __init__(self,args,agent_num=None,lr=None,seed=None):
        self.args=args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.open_grad_flag=True

        if lr is None:
            self.lr=args.lr
        else:
            self.lr =lr

        if agent_num is None:
            self.agent_num = args.player_num
        else:
            self.agent_num = agent_num

        self.loss_list=[]



        self.model = example_payment_Net(agent_number=self.agent_num, device=self.device)

        if seed is not None:
            set_seed(seed=seed)  # if set seed

        self.loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss() #nn.MSELoss()

        self.loss_fn2 = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss() #nn.MSELoss()


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.to(self.device)

    def open_grad(self):
        print('-----open payment net gradient mode!!-----')
        self.open_grad_flag=True
        self.model.train()

    def close_grad(self):
        self.open_grad_flag=False
        self.model.eval()

    def compute_payment(self,state, possible_agents,winner=0):

        #payment = {agent: 0 for agent in possible_agents}

        # state = biddings

        biddings = [state[agt] for agt in possible_agents]

        #convert to tensor
        convert_state = self.convert_to_tensor(biddings)
        if self.open_grad_flag:
            logist = self.model(convert_state)
        else:
            with torch.no_grad():
                logist = self.model(convert_state)

        #winner_id = possible_agents.index(winner)

        #payment = logist[winner_id] *state[winner]

        if self.open_grad_flag:
            payment = {agent: logist[possible_agents.index(agent)] * -1 * state[agent] for agent in possible_agents}
        else:
            payment = {agent: self.convert_to_float(logist[possible_agents.index(agent)]*-1*state[agent]) for agent in possible_agents}


        #pp=self.convert_to_float(payment)

        return payment

    def convert_to_tensor(self,input_batch):
       if not isinstance(input_batch, torch.Tensor):
                input_batch=torch.tensor(input_batch)

       return input_batch.float().to(self.device)

    def convert_to_float(self,input_batch):
        return float(input_batch.cpu())

    def plot_loss(self,output_path):
        import matplotlib.pyplot as plt
        plt.ylabel('bid/valuation')
        plt.plot([x  for x in range(len(self.loss_list))],self.loss_list)
    def print_loss(self):
        cnt=0
        for data in self.loss_list:
            print('---epoch' + str(cnt) +' loss is '+ str(data) +' ---')
            cnt+1

    def model_train(self,dataset,agent_name_list,batchsize=16):
        self.optimizer.zero_grad()  # optimizer init

        total_revenue=0
        agent_num = len(agent_name_list)

        for i in range(batchsize):

            round_revenue=dataset[agent_name_list[0]]['payment'][i]


            for agent_index in range(agent_num-1):
                agent_name = agent_name_list[agent_index+1]
                round_revenue =torch.add(round_revenue,dataset[agent_name]['payment'][i]) # e.g: -3, -4 with grad
            #deal with round revenue
            if i==0:
                total_revenue = round_revenue
            else:
                total_revenue = torch.add(total_revenue,round_revenue)

        loss = total_revenue /batchsize
        loss.backward()

        print('----------model training ---------')
        print(' --> the training loss(mse)  with batch size ' + str(
            batchsize) + ' is ' + str(loss.item()))

        print('----------end this turn :model training ---------')
        #print('----------model training ---------')

        self.optimizer.step()

        self.loss_list.append(loss.item())
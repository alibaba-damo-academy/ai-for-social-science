import torch
import os
import  random
import numpy as np
from torch import nn


# denote as second price network
def set_seed(args, seed=0):
    # Set seed for result reproducibility
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class example_allocation_Net(nn.Module):
    def __init__(self,agent_number=4,device=torch.device('cpu')):
        super().__init__()

        self.agent_num=agent_number
        self.device=device

        self.model = nn.Sequential(
            nn.Linear(np.prod(self.agent_num), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(self.agent_num)),
            nn.Softmax(dim=1)
        )

    def forward(self,state):
        # compute q value

        #get the softmax allocation possibility

        logits = self.model(state)

        # argmax q value
        max_p ,winner = logits.max(dim=-1)

        # batchsize *1

        return logits,max_p, winner



class deep_allocation(object):

    def __init__(self,args,agent_num=None,lr=None,seed=None):
        self.args=args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.open_grad_flag = True

        if lr is None:
            self.lr=args.lr
        else:
            self.lr =lr

        if agent_num is None:
            self.agent_num = args.player_num
        else:
            self.agent_num = agent_num




        self.model = example_allocation_Net(agent_number=self.agent_num, device=self.device)

        if seed is not None:
            set_seed(seed=seed)  # if set seed

        self.loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss() #nn.MSELoss()

        self.loss_fn2 = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss() #nn.MSELoss()


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.to(self.device)

    def open_grad(self):
        self.open_grad_flag = True
        self.model.train()

    def close_grad(self):
        self.open_grad_flag = False
        self.model.eval()

    def compute_allocation(self,state, possible_agents,reserve_price=0):

        # state = biddings
        biddings = [state[agt] for agt in possible_agents]

        convert_state = self.convert_to_tensor(biddings)

        if self.open_grad_flag:
            logist,max_p,winner_id = self.model(convert_state)
        else:
            with torch.no_grad():
                logist, max_p, winner_id = self.model(convert_state)



        winner = possible_agents[winner_id]


        return winner

    def convert_to_tensor(self, input_batch):
        if not isinstance(input_batch, torch.Tensor):
            input_batch = torch.tensor(input_batch)

        return input_batch.float().to(self.device)

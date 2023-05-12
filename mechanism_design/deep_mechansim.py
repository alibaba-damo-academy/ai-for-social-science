import torch

import sys
sys.path.append('../')

from env.policy import *

class deep_policy(Policy):

    def __init__(self,allocation_mode='deep', payment_mode='deep',action_mode='same',args=None):
        super(deep_policy).__init__()



tmp =deep_policy(allocation_mode='deep')
agents=['agent_1','agent_2']
tmp.assign_agent(agents)
print(tmp)
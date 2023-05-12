import numpy as np
import random
from copy import deepcopy
import sys

sys.path.append('../')
from agent_utils.signal import *
from agent_utils.valuation_generator import *


class base_private_Signal_generator(object):
    def __init__(self, dtype='Discrete', default_feature_dim=1, default_lower_bound=0, default_upper_bound=10,
                 default_generation_method='uniform', default_value_to_signal=0, agent_name_list=[]):

        # only generator signal as a vector or a list
        # [s1,s2,s3..sn]

        super(base_private_Signal_generator, self).__init__()

        self.dtype = dtype
        self.default_feature_dim = default_feature_dim
        self.default_lower_bound = default_lower_bound
        self.default_upper_bound = default_upper_bound
        self.epoch = 0
        self.default_generation_method = default_generation_method
        # self.signal_realization = None
        self.default_value_to_signal = default_value_to_signal
        self.agent_name_list = agent_name_list

        self.agent_private_signal_list = {agent: None for agent in self.agent_name_list}

    def generate_default_iid_signal(self):

        for agent in self.agent_name_list:
            iid_signal = deepcopy(Signal(dtype='Discrete', feature_dim=self.default_feature_dim,
                                         lower_bound=self.default_lower_bound,
                                         upper_bound=self.default_upper_bound,
                                         generation_method=self.default_generation_method,
                                         value_to_signal=self.default_value_to_signal,
                                         public_signal_asym=0
                                         ))
            self.agent_private_signal_list[agent] = iid_signal
            self.agent_private_signal_list[agent].signal_init()

    def set_agent_signal(self, agent_name, signal):
        if self.check_avaliable_agent_name(agent_name):
            self.agent_private_signal_list[agent_name] = signal
            print('----- set signal success on agent ' + str(agent_name) + ' ----')

        else:
            # not in the agent name list
            print('agent ' + str(agent_name) + ' not in the agent list below --->')
            print(self.agent_name_list)
            print('----- set signal failed ----')

    def reset_agent_list(self, agent_name_list=None, re_init=True):

        if agent_name_list is None:
            print('---try to reset agent name list to None , please input agent name list ')
            return

        self.agent_name_list = agent_name_list
        if re_init:
            self.generate_default_iid_signal()

    def check_avaliable_agent_name(self, agent_name):
        return agent_name in self.agent_name_list

    def generate_agt_signal(self, agent_name, init=True, data_type='int', upper_bound=None):

        if not self.check_avaliable_agent_name(agent_name):
            # not appear in list
            return 0

        signal = self.agent_private_signal_list[agent_name]

        signal.generate_signal(init=init, data_type=data_type, upper_bound=upper_bound)

        return 1


    def get_latest_signal(self,agent_name):
        if not self.check_avaliable_agent_name(agent_name):
            # not appear in list
            return 0

        signal = self.agent_private_signal_list[agent_name]

        return  signal.get_whole_signal_realization()
import numpy as np
import random
from copy import deepcopy
import sys
sys.path.append('../')
from agent_utils.signal import *
from agent_utils.valuation_generator import *


class platform_Signal_v1(Signal):
    def __init__(self, dtype='Discrete', feature_dim=1, lower_bound=0, upper_bound=10,
    generation_method='uniform',value_to_signal=0, public_signal_asym = 0):

        #only generator signal as a vector or a list
        #[s1,s2,s3..sn]

        super(platform_Signal_v1, self).__init__()

        self.dtype = dtype
        self.feature_dim = feature_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.epoch = 0
        self.generation_method = generation_method
        self.signal_realization = None
        self.value_to_signal=value_to_signal
        self.public_signal_asym = public_signal_asym


    def signal_init(self):
        self.signal_realization = [None] * self.feature_dim

    def generate_signal(self, init=True, data_type='int',upper_bound=None):
        if init:
            self.signal_init()

        if upper_bound is not None:
            cur_upper_bound=upper_bound
        else:
            cur_upper_bound=self.upper_bound

        if self.dtype == "Discrete":
            if self.generation_method == 'uniform':
                for i in range(self.feature_dim):
                    rnd_signal = np.random.uniform(self.lower_bound, cur_upper_bound)  # [low , high]
                    if data_type == 'int':
                        rnd_signal = int(rnd_signal)
                    self.signal_realization[i] = rnd_signal

        return

    def get_whole_signal_realization(self):
        return self.signal_realization

    def get_partial_signal_realization(self, observed_dim_list=[]):
        if observed_dim_list is None:
            observed_dim_list = []  # observe None signal

        # when user observe partial signal
        partial_signal = deepcopy(self.signal_realization)
        for dim in range(self.feature_dim):
            if dim not in observed_dim_list:
                partial_signal[dim] = None  # not observe the signal
        return partial_signal  # the same dim as the whole signal with None


def weighted_sum(signal,weights):
    if weights is None:
        print('please input the weights for weighted sum')
        return -1

    value=0
    for i in len(signal):
        value+=signal[i]*weights[i]
    return value



def platform_public_signal_to_value(signal,func='sum',weights=None):
    #
    # denotes the function for signal->value [v=f(s)]
    #
    if signal is None or len(signal)==0:
        return 0

    if func =='sum':
        return sum(signal)
    if func =='avg':
        return sum(signal) / len(signal)


    if func == 'weighted':
        return weighted_sum(signal,weights)

def private_signal_to_value(signal,func='sum',weights=None):
    #
    # denotes the function for signal->value [v=f(s)]
    #
    if signal is None or len(signal) == 0:
        return 0

    if func == 'sum':
        return sum(signal)
    if func == 'avg':
        return sum(signal) / len(signal)

    if func == 'weighted':
        return weighted_sum(signal, weights)



class platform_Signal_v2_rnd(Signal):
    def __init__(self, dtype='Discrete', feature_dim=1, lower_bound=0, upper_bound=10,
    generation_method='uniform',value_to_signal=0, public_signal_asym = 0):

        #only generator signal as a vector or a list
        #[s1,s2,s3..sn]

        super(platform_Signal_v2_rnd, self).__init__()

        self.dtype = dtype
        self.feature_dim = feature_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.epoch = 0
        self.generation_method = generation_method
        self.signal_realization = None
        self.value_to_signal=value_to_signal
        self.public_signal_asym = public_signal_asym

        #added
        self.hidden_signal =None


    def signal_init(self):
        self.signal_realization = [None] * self.feature_dim

    def generate_signal(self, init=True, data_type='int',upper_bound=None):
        if init:
            self.signal_init()

        if upper_bound is not None:
            cur_upper_bound=upper_bound
        else:
            cur_upper_bound=self.upper_bound

        p=np.random.random()
        self.hidden_signal=None


        if self.dtype == "Discrete":
            if self.generation_method == 'uniform':
                for i in range(self.feature_dim):
                    rnd_signal = np.random.uniform(self.lower_bound, cur_upper_bound )  # [low , high]
                    if data_type == 'int':
                        rnd_signal = int(rnd_signal)
                    self.signal_realization[i] = rnd_signal

        if p<0.5:
            #then hidden
            self.hidden_signal=deepcopy(self.signal_realization)
            self.signal_realization=[0]*self.feature_dim


        return

    def get_whole_signal_realization(self):
        return self.signal_realization

    def get_partial_signal_realization(self, observed_dim_list=[]):
        if observed_dim_list is None:
            observed_dim_list = []  # observe None signal

        # when user observe partial signal
        partial_signal = deepcopy(self.signal_realization)
        for dim in range(self.feature_dim):
            if dim not in observed_dim_list:
                partial_signal[dim] = None  # not observe the signal
        return partial_signal  # the same dim as the whole signal with None


    def get_hidden_signal(self):
        return self.hidden_signal
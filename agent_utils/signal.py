import numpy as np
import random
from copy import deepcopy


class Signal(object):
    def __init__(self, dtype='Discrete', feature_dim=1, lower_bound=0, upper_bound=10,
    generation_method='uniform',value_to_signal=0, public_signal_asym = 0):
        self.dtype = dtype
        self.feature_dim = feature_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.epoch = 0
        self.generation_method = generation_method
        self.signal_realization = None
        self.value_to_signal=value_to_signal
        self.public_signal_asym = public_signal_asym


    def adjust_signal(self, dim, given_signal):
        if self.signal_realization is not None and dim < len(self.signal_realization):
            self.signal_realization[dim] = given_signal
        else:
            print('out of signal range in dim ' + str(dim))

    def signal_init(self):
        self.signal_realization = [None] * self.feature_dim

    def generate_signal(self, init=True, data_type='int',upper_bound=None):
        if init:
            self.signal_init()

        if self.value_to_signal==1:
            if upper_bound is None:
                return # to meet the standard procedure in main.py where signal generate before value
                #public_signal_generator.generate_signal(data_type='int')  #update the signal

            if self.generation_method == 'uniform':
                for i in range(self.feature_dim):
                    rnd_signal = np.random.uniform(0, 2*upper_bound + 1)  # [low , high] | 0-2*v
                    if data_type == 'int':
                        rnd_signal = int(rnd_signal)

                    self.signal_realization[i] = rnd_signal
                if self.public_signal_asym: 
                    for i in range(self.feature_dim):
                        rnd_signal = np.random.uniform((1/self.feature_dim*i)*upper_bound, (2-1/self.feature_dim*i)*upper_bound + 1)  # [low , high] | 0-2*v
                        if data_type == 'int':
                            rnd_signal = int(rnd_signal)

                        self.signal_realization[i] = rnd_signal


        elif self.dtype == "Discrete":
            if self.generation_method == 'uniform':
                for i in range(self.feature_dim):
                    rnd_signal = np.random.uniform(self.lower_bound, self.upper_bound + 1)  # [low , high]
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



class KL_Signal(object):
    def __init__(self, dtype='Discrete', feature_dim=1, lower_bound=0, upper_bound=10, generation_method='uniform',value_to_signal=0,
                 value_generator_mode='single'

                 ):
        self.dtype = dtype
        self.feature_dim = feature_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.epoch = 0
        self.generation_method = generation_method
        self.signal_realization = None
        self.value_to_signal=value_to_signal

        self.value_generator_mode=value_generator_mode

        self.shared_signal=None

    def adjust_signal(self, dim, given_signal):
        if self.signal_realization is not None and dim < len(self.signal_realization):
            self.signal_realization[dim] = given_signal
        else:
            print('out of signal range in dim ' + str(dim))

    def signal_init(self):
        self.signal_realization = [None] * self.feature_dim
        self.shared_signal=None

    def generate_signal(self, init=True, data_type='int',upper_bound=None):

        if init:
            self.signal_init()

        if self.value_to_signal==1:
            if upper_bound is None:
                return # to meet the standard procedure in main.py where signal generate before value
                #public_signal_generator.generate_signal(data_type='int')  #update the signal

            if self.generation_method == 'uniform':
                for i in range(self.feature_dim):
                    rnd_signal = np.random.uniform(0, 2*upper_bound + 1)  # [low , high] | 0-2*v
                    if data_type == 'int':
                        rnd_signal = int(rnd_signal)

                    self.signal_realization[i] = rnd_signal



        elif self.dtype == "Discrete":
            # assign upper bound for future use
            if upper_bound is None:
                upper_bound=self.upper_bound


            if self.generation_method == 'uniform':
                for i in range(self.feature_dim):
                    rnd_signal = np.random.uniform(self.lower_bound, upper_bound + 1)  # [low , high]
                    if data_type == 'int':
                        rnd_signal = int(rnd_signal)

                    self.signal_realization[i] = rnd_signal

                #apply shared signal T
                self.shared_signal = np.random.uniform(self.lower_bound, self.upper_bound + 1)  # [low , high]
                if data_type == 'int':
                    self.shared_signal = int(self.shared_signal)

                for i in range(self.feature_dim):

                    if self.value_generator_mode=='single_u': #i>0 and self.value_generator_mode=='single_u':
                        continue
                    else:
                        self.signal_realization[i] +=self.shared_signal

                return

    def get_shared_signal(self):
        return self.shared_signal

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





def signal_reshape_dim(partial_signal):  # rebuild partial signal into non-None signal

    if partial_signal is None:
        return None

    rebuild_signal = []
    for dim in range(len(partial_signal)):
        if partial_signal[dim] is not None:
            rebuild_signal.append(partial_signal[dim])

    if len(rebuild_signal) == 0:
        return None
    else:
        return rebuild_signal


def max_signal_value(args):
    highest_value = 0
    if 'kl' in args.folder_name:
        each_sigal_value_range=2*args.public_signal_range

        # suppose 1 signal yet
        highest_value = each_sigal_value_range
        if args.agt_obs_public_signal_dim >=2:
            print('KL exp exceed dim 2 as we do not extend yet')
            print(1/0)

    else:
        each_sigal_value_range = args.public_signal_range #[0-K]
        for j in range(args.agt_obs_public_signal_dim):
            highest_value += each_sigal_value_range * ((each_sigal_value_range + 1) ** j)




    return highest_value


def signal_decode(args, state):
    signal = []
    for j in range(args.agt_obs_public_signal_dim):
        laten=(args.public_signal_range + 1)

        tmp_signal = state % laten
        state =int(state/laten)

        signal.append(tmp_signal)

    return signal


def signal_to_value_sum(signal):
    value = sum(signal)  # directly sum

    return value

def signal_to_value_mean(signal,mean_len=None):
    if mean_len is None:
        value = sum(signal) / len(signal)
    else:
        value = sum(signal) / mean_len
    return value
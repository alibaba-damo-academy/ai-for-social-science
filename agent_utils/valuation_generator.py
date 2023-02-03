from .signal import *


class base_valuation_generator(object):
    def __init__(self):
        self.generator_method = None
        self.common_signal_realization = None
        self.private_signal_realization = None
        self.signal_weight = None

    def value_generate(self):
        return

    def update_private_signal(self, private_signal_realization):
        self.private_signal_realization = private_signal_realization

    def update_common_signal(self, common_signal_realization):
        self.common_signal_realization = common_signal_realization

    def update_signal_weight(self, weight):
        self.signal_weight = weight

    def generate_value(self):
        # function from signal -> value  function
        # can become signal ->distribution -> value

        return

    def get_private_signal(self):
        return self.private_signal_realization

    def get_public_signal(self):
        return self.common_signal_realization

    def generate_signal_weight(self, signal_dim=1):
        return


class private_value_generator(base_valuation_generator):
    # in this demo only one private value generator with multiple private value generator
    # different users observe different perspectives of the private value
    # also can apply multiple private value generator

    # this class is demoed for generator and project from signal realization to value
    def __init__(self, signal_generator=None, value_generator_mode='sum'):
        super(private_value_generator, self).__init__()
        self.private_signal_generator = signal_generator
        self.value_generator_mode = value_generator_mode
        self.generate_signal_weight(signal_dim=self.private_signal_generator.feature_dim)

        self.latest_private_gnd_value = None
        self.step = 0

    def get_private_gnd_value(self):
        return self.latest_private_gnd_value

    def generate_signal_weight(self, signal_dim=1):
        ## apply weight dim
        weight = []

        for i in range(signal_dim):
            weight.append(i + 1)  # weight as i
        # or can apply other weight

        self.signal_weight = weight

    def next_step(self):
        # Not used in the current implementation 
        if self.private_signal_generator is not None:
            # first generate the whole signal from the platform view
            self.private_signal_generator.generate_signal()
            whole_signal_realization = self.private_signal_generator.get_whole_signal_realization()

            self.update_private_signal(common_signal_realization=whole_signal_realization)

            self.step += 1

    def get_partial_signal(self, obs=None):

        if self.value_generator_mode in ['add','mean']:
            # mode ==add :
            # agent observe the true signal ,but can only observe the part of the dim from the whole signal

            return self.private_signal_generator.get_partial_signal_realization(observed_dim_list=obs)
        else:
            # other mode can be added
            # such as agent can observe all the signal, but the precise of the signal is diverse

            return None

    def generate_value(self, obs=None, gnd=False, weighted=False):
        # obs is the list of which dim agent can observed

        # function from signal | partial signal -> detailed value

        signal = None

        # build partial view
        if gnd:
            signal = self.get_private_signal()
        else:
            signal = self.get_partial_signal(obs=obs)

        # then from signal -> value

        if self.value_generation_mode in ['add','mean']:

            reshaped_signal = signal_reshape_dim(signal)
            # signal is a random variable and the value is a fixed value if the signal is determined
            # signal -> value
            if reshaped_signal is None:
                value = 0
            else:
                if weighted:
                    value = 0
                    for i in range(len(signal)):
                        if signal[i] is not None:
                            if self.signal_weight[i] is not None:
                                value += signal[i] * self.signal_weight[i]
                            else:
                                value += signal[i]
                else:

                    value = sum(reshaped_signal)  # directly sum
                # if self.value_generation_mode == 'mean':
                #     if weighted: 
                #         value = value/sum(self.signal_weight)
                #     else: 
                #         value = value/len(signal)
            return value


class public_value_generator(base_valuation_generator):
    # in this demo only one public value generator with multiple private value generator
    # different users observe different perspectives of the public value
    # also can apply multiple public value generator

    # this class is demoed for generator and project from signal realization to value
    def __init__(self, signal_generator=None, value_generator_mode='sum',valuation_range=0):
        super(public_value_generator, self).__init__()
        self.public_signal_generator = signal_generator
        self.value_generator_mode = value_generator_mode
        self.generate_signal_weight(signal_dim=self.public_signal_generator.feature_dim)


        self.function_list=['sum','mean','single','single_u']


        self.valuation_range=valuation_range

        self.latest_public_gnd_value = None
        self.step = 0

    def get_last_public_gnd_value(self):
        return self.latest_public_gnd_value

    def record_public_gnd_value(self,public_gnd_value):
        self.latest_public_gnd_value=public_gnd_value

    def generate_signal_weight(self, signal_dim=1):
        ## apply weight dim
        weight = []

        for i in range(signal_dim):
            weight.append(i + 1)  # weight as i
            # weight.append(1) # uniform weight
        # or can apply other weight

        self.signal_weight = weight

    def next_step(self, generated_signal_realization=None):
        # Generate the public signal realization at the start of the auction
        if generated_signal_realization is None:
            if self.public_signal_generator is not None:
                # first generate the whole signal from the platform view
                self.public_signal_generator.generate_signal()
                whole_signal_realization = self.public_signal_generator.get_whole_signal_realization()
        else:
                whole_signal_realization =generated_signal_realization
        self.update_common_signal(common_signal_realization=whole_signal_realization)

        self.step += 1

    def get_partial_signal(self, obs=None):

        if self.value_generator_mode in self.function_list:
            # mode ==add :
            # agent observe the true signal ,but can only observe the part of the dim from the whole signal

            return self.public_signal_generator.get_partial_signal_realization(observed_dim_list=obs)
        else:
            # other mode can be added
            # such as agent can observe all the signal, but the precise of the signal is diverse

            return None

    def generate_value(self, obs=None, gnd=False, weighted=False,obs_list=None):
        # obs is the list of which dim agent can observed
        # function from signal | partial signal -> detailed value
        signal = None

        # build partial view
        if gnd:
            signal = self.get_public_signal()
        elif obs is None:
            signal = self.get_partial_signal(obs=obs_list) #load from obs dim list
        else:
            signal = obs

        # then from signal -> value

        if self.value_generator_mode in self.function_list:
            # Generate the public value from the public signal
            reshaped_signal = signal_reshape_dim(signal)
            # signal is a random variable and the value is a fixed value if the signal is determined
            # signal -> value
            if reshaped_signal is None:
                value = 0
            else:
                if weighted:
                    value = 0
                    for i in range(len(signal)):
                        if signal[i] is not None:
                            if self.signal_weight[i] is not None:
                                value += signal[i] * self.signal_weight[i]
                            else:
                                value += signal[i]
                else:
                    if self.value_generator_mode =='sum':
                        value = signal_to_value_sum(reshaped_signal)  # directly sum
                    elif self.value_generator_mode =='mean':
                        value = signal_to_value_mean(reshaped_signal) # see function define
                    elif self.value_generator_mode =='single' or self.value_generator_mode =='single_u' :
                        value =reshaped_signal[0] #assign the first agent to the all-known agent

                    else:
                        print('current mode is ?' + str(self.value_generator_mode))
                        value = signal_to_value_sum(reshaped_signal)  # directly sum
            return value

        elif self.value_generator_mode =='fixed':
            if gnd: #from the beginning of the observation, generate the ground truth public value
                value = int(np.random.uniform(0, self.valuation_range + 1))  # [low , high]
                return value
            else:
                # Provide the same public value generated at the start of each auction
                return self.latest_public_gnd_value
            

from .normal_agent import *
import random


class signal_agent(base_agent):
    """
    signal agent generates its action based on certain greedy 'self.algorithm'.
    his public signal and private signal devote to his true value
    """

    def __init__(self, agent_name='', obs_public_signal_dim=2, max_public_dim=5, public_signal_range=10,
                 private_signal_generator=None):
        super(signal_agent, self).__init__(agent_name)

        # public signal
        self.obs_public_signal_dim = obs_public_signal_dim  # how much dim of the public signal is observed

        self.public_signal_dim_list = None
        self.max_public_dim = max_public_dim
        self.public_signal_range = public_signal_range  # user observed maximal signal upper bound

        # private signal
        self.private_signal_generator = private_signal_generator

        # other profile
        self.last_encode_signal = None
        self.last_public_signal_realization = None

    def set_public_signal_dim(self, observed_dim_list=[], rnd=False):
        if rnd or len(observed_dim_list) < self.obs_public_signal_dim:
            dim_mask = [i for i in range(self.max_public_dim)]
            random.shuffle(dim_mask)
            self.public_signal_dim_list = dim_mask[:self.obs_public_signal_dim]
        else:
            self.public_signal_dim_list = observed_dim_list[:self.obs_public_signal_dim]

        # sort the list
        self.public_signal_dim_list.sort()

        return

    def get_public_signal_dim_list(self):
        return self.public_signal_dim_list

    def receive_partial_obs_true_value(self, true_value):
        self.record_true_value(true_value)
        return

    def generate_true_value(self, method='uniform'):

        # Generate the true value given the true value

        # should not exist generate true value method if no private value
        if self.private_signal_generator is None:
            return
        print('no method found')
        return

    def generate_action(self, observation, public_signal_realization=None):

        # Generate the action given its observed true value of the item

        # for the first version for temp eric:
        public_signal_realization = observation

        # merge the private signal and the public signal realization

        state = self.encode_signal(public_signal=public_signal_realization,
                                   private_signal=None)

        # here
        encoded_action = self.algorithm.generate_action(obs=state)

        # Decode the encoded action
        action = encoded_action % (self.algorithm.bidding_range)

        # record
        self.record_action(action)
        self.last_public_signal_realization = public_signal_realization
        self.last_encode_signal = state

        return action

    def update_policy(self, state, reward,done=False):

        last_action = self.get_latest_action()
        encoded_action = last_action + self.last_encode_signal * (self.algorithm.bidding_range)

        self.algorithm.update_policy(encoded_action, reward=reward)

        self.record_reward(reward)

        return

    def encode_signal(self, public_signal, private_signal=None):
        """
        from signal to a value (or state)
        encode for the algorithm optimization
        in signal , None = not observed
        in state ,  v1: no deal with not observed signal = no idea about further information on non-observed signal
        """
        state = 0

        if private_signal is None:
            cnt = 0
            for obs_dim in (self.public_signal_dim_list):
                obs_signal = public_signal[obs_dim]
                # check if it is none:
                if obs_signal is None:
                    print('exist None in ' + str(self.agent_name) + 'signal_dim is ' + str(obs_signal))
                else:
                    state += obs_signal * ((self.public_signal_range + 1) ** cnt) #[0-H]
                cnt += 1

        return state

    def encode(self, last_state):
        """
        Encode the last state (a value,action tuple) as a single number.

        Encode the output the RL agent so that the RL algorithm can run on different interface.
        For example, change the price, or price higher than other agent.
        """

        [last_true_value, last_action] = last_state  # [1-H, 0-H-1]
        res = (last_true_value) * (self.algorithm.bidding_range) + last_action  # 观测1-10块钱，但是可以出价0.1- 10.0 所以100个切分

        return res

    def test_policy(self, state):

        encoded_action = self.algorithm.generate_action(obs=state, test=True)

        action = encoded_action % (self.algorithm.bidding_range)

        return action

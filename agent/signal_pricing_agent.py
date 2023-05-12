import random
from rl_utils.solver import *

from .custom_agent import *


class simple_signal_greedy_agent(custom_agent):

    def __init__(self, args=None, agent_name='', obs_public_signal_dim=2, public_signal_range=10,private_signal_range =10,
                 private_signal_generator=None):
        super(simple_signal_greedy_agent, self).__init__()

        self.args = args
        self.agent_name=agent_name

        # public signal
        self.obs_public_signal_dim = obs_public_signal_dim  # how much dim of the public signal is observed

        self.public_signal_dim_list = None

        self.public_signal_range = public_signal_range  # user observed maximal signal upper bound
        self.private_signal_range = private_signal_range

        # private signal
        self.private_signal_generator = private_signal_generator


        # other profile
        self.last_encode_signal = None
        self.last_public_signal_realization = None

        self.signal_history=[]



        # self.set_algorithm(
        #     EpsilonGreedy_auction(bandit_n=(highest_value + 1) * self.args.bidding_range,
        #                           bidding_range=self.args.bidding_range, eps=0.01,
        #                           start_point=int(self.args.exploration_epoch),
        #                           # random.random()),
        #                           overbid=self.args.overbid, step_floor=int(self.args.step_floor),
        #                           signal=self.args.public_signal
        #                           )  # 假设bid 也从分布中取
        # )


        # add by Xue
        self.signal_history_ppo=np.array([])
        self.action_history_ppo=np.array([])
        self.prob_history_ppo=np.array([])
        self.value_history_ppo=np.array([])
        self.reward_histoy_ppo = np.array([])
        self.dist_history_ppo=[]


    def record_signal(self, signal):
        self.signal_history.append(signal)


        # add by Xue
    def record_signal_ppo(self, signal=None,action=None,prob=None,value=None,reward=None,dist=None):
        if signal is not None:
            self.signal_history_ppo = np.append(self.signal_history_ppo, signal)
        if action is not None:
            self.action_history_ppo = np.append(self.action_history_ppo, action)
        if prob is not None :
            self.prob_history_ppo = np.append(self.prob_history_ppo, prob)
        if value is not None:
            self.value_history_ppo = np.append(self.value_history_ppo, value)
        if reward is not None:
            self.reward_histoy_ppo = np.append(self.reward_histoy_ppo, reward)
        if dist is not None:
            self.dist_history_ppo.append(dist)



    def receive_observation(self, args, budget, agent_name, agt_idx,
                            extra_info, observation, public_signal,
                            #true_value_list, action_history, reward_history, allocation_history
                            ):
        obs = dict()
        obs['observation'] = observation['observation']

        if budget is not None:
            budget.generate_budeget(user_id=agent_name)
            obs['budget'] = budget.get_budget(user_id=agent_name)  # observe the new budget

        if args.communication == 1 and agt_idx in args.cm_id and extra_info is not None:
            obs['extra_info'] = extra_info[agent_name]['others_value']

        if args.public_signal:
            obs['public_signal'] = public_signal

        # obs['true_value_list'] = true_value_list
        # obs['action_history'] = action_history
        # obs['reward_history'] = reward_history
        # obs['allocation_history'] = allocation_history

        self.record_obs(obs)

        return obs


    def set_public_signal_dim(self, observed_dim_list=[]):

        self.public_signal_dim_list = observed_dim_list[:self.obs_public_signal_dim]

        # sort the list
        self.public_signal_dim_list.sort()

        return

    def get_public_signal_dim_list(self):
        return self.public_signal_dim_list

    def receive_partial_obs_true_value(self, true_value):
        self.record_true_value(true_value)
        return


    def generate_action(self, observation, public_signal_realization=None):

        # Generate the action given its observed true value of the item

        # for the first version for temp eric:
        public_signal_realization = observation['public_signal']

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

    def update_policy(self, obs, reward,done=False):

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
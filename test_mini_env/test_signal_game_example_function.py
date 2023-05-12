from copy import deepcopy
import numpy as np
import sys
sys.path.append('../')
from env.multi_dynamic_env import *
from mini_env.base_signal_mini_env import *
from env.self_play_env import *
from agent.normal_agent import *
from agent.custom_agent import *
from rl_utils.solver import *
from rl_utils.reward_shaping import *
from rl_utils.budget import *
from rl_utils.communication import *
from rl_utils.util import *
from plot.plot_util import *
from plot.print_process import *
from env.env_record import *
from env.policy import *
from agent_utils.action_encoding import *
from agent_utils.action_generation import *
from agent_utils.agent_updating import *
from agent_utils.obs_utils import *
from agent_utils.signal import *
from agent_utils.valuation_generator import *

from agent.agent_generate import *

from pricing_signal.plat_deliver_signal import *


def get_custom_agent_name():
    return 'player_3'


def get_custom_rl_agent(args):

    # in signal game
    custom_agt = signal_agent(agent_name=get_custom_agent_name(),
                 obs_public_signal_dim=args.agt_obs_public_signal_dim,
                 max_public_dim=args.public_signal_dim,
                 public_signal_range=args.public_signal_range,
                 private_signal_generator=None)


    #custom_agent(get_custom_agent_name(), highest_value=args.valuation_range - 1)

    return custom_agt

def get_custom_algorithm(args):
    return EpsilonGreedy_auction(bandit_n=args.valuation_range * args.bidding_range,
                                 bidding_range=args.bidding_range, eps=0.01,
                                 start_point=int(args.exploration_epoch),
                                 # random.random()),
                                 overbid=args.overbid, step_floor=int(args.step_floor),
                                 signal=args.public_signal)  # 假设bid 也从分布中取


def generate_action(self, obs):
    # print('self', self)
    # print('obs', obs)
    encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

    # Decode the encoded action
    action = encoded_action % (self.algorithm.bidding_range)

    self.record_action(action)
    return action
    # return 1


def update_policy(self, obs, reward, done):
    last_action = self.get_latest_action()
    if done:
        true_value = self.get_latest_true_value()
    else:
        true_value = self.get_last_round_true_value()
    encoded_action = last_action + true_value * (self.algorithm.bidding_range)
    self.algorithm.update_policy(encoded_action, reward=reward)

    self.record_reward(reward)
    return


def receive_observation(self, args, budget, agent_name, agt_idx,
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
    return obs


def platform_public_signal_to_value(self,signal, func='sum', weights=None):
    #

    value = int( sum(signal) / 2)
    #print('new signal func')
    return value


def platform_public_signal_to_value_mode3(self,signal, func='sum', weights=None):
    #
    #print('new signal func')
    #print(self)
    if max(signal) ==0 :

        signal = self.saved_signal
    if signal is None:
        return 0

    value = int( sum(signal))
    #print('new signal func')

    return value

def test_update_env_public_signal(self):

    if self.public_signal_generator is None:
        # not exist signal generator
        print('not exist signal generator, skip signal updating ')
        return

    #print('test!')

    self.last_public_signal = deepcopy(self.public_signal_generator.get_whole_signal_realization())
    # receive the public signal results and saved

    self.public_signal_generator.generate_signal(data_type='int')  # update the signal

    #add save current random value


    #hidden

    self.saved_signal = self.public_signal_generator.get_hidden_signal()



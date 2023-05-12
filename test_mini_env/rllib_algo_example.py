from copy import deepcopy
import numpy as np
import os
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
from examples.customed_env_auction_deep import *


from ray.tune.logger import pretty_print
import ray
from ray import air, tune
from ray.rllib.algorithms.pg import (
    PG,
    PGTF2Policy,
    PGTF1Policy,
    PGTorchPolicy,
)
import numpy as np
from ray.tune.registry import get_trainable_cls
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples.policy.rock_paper_scissors_dummies import (
    BeatLastHeuristic,
    AlwaysSameHeuristic,
)
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

from pathlib import Path

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

def get_custom_agent_name():
    return 'player_3'

def encoded_input_emb(action,action_space_num=100):
    res=torch.zeros(1,action_space_num+1)
    res[0][action]=1
    return res

## rllib settings
def build_multi_policy_name(agent_num =2,framework=None):
    res={}
    for i in range(agent_num):
        name = 'player_'+str(i)
        res[name]=PolicySpec(
                    config={
                        "framework": framework,
                    })
    return res

def build_single_policy(agent_name='',framework=None):
    res={}
    res[agent_name]=PolicySpec(
                    config={
                        "framework": framework,
                    })
    return res

def RLlib_algorithm_config(args,agt_num=2,framework="torch"):
    config = {
        "env": "second_auction",
        "framework": framework,
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "3")),
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 10,
        "train_batch_size": 200,
        "metrics_num_episodes_for_smoothing": 200,
        "multiagent": {
            "policies": build_multi_policy_name(agent_num=agt_num,framework=framework),
            #build_policy_name(agent_num=args.player_num, args=args),
            #     {
            #     "player_0": PolicySpec(
            #             config={
            #                 "framework": args.framework,
            #             }),
            #     "player_1": PolicySpec(
            #             config={
            #                 "framework": args.framework,
            #             }),
            # },
            "policy_mapping_fn": lambda agent_id: agent_id,
        }
    }

    return config

##

def get_custom_algorithm_in_rllib(args,algorithm='DQN'):

    #get self_play env from mini_env
    # use mini_env to set a empty env used for self_play

    #suppuse use a single signal -> value second env algorithem
    assumed_player_num=2
    training_env_step=1000
    self_play_env = customed_env(player_num=assumed_player_num,
                                 action_space_num=args.bidding_range,
                                 second_price=(args.mechanism =='second_price'),
                                 env_iters =training_env_step)
    self_play_env.reset()
    register_env("second_auction", lambda config: PettingZooEnv(self_play_env))

    cls = get_trainable_cls(algorithm) if isinstance(algorithm, str) else algorithm

    config=RLlib_algorithm_config(args,agt_num=assumed_player_num)

    algo = cls(config=config)
    # first train through self play env
    algo.train()

    first_name='player_0'


    return algo.get_policy(first_name)
        # #EpsilonGreedy_auction(bandit_n=args.valuation_range * args.bidding_range,
        #                          bidding_range=args.bidding_range, eps=0.01,
        #                          start_point=int(args.exploration_epoch),
        #                          # random.random()),
        #                          overbid=args.overbid, step_floor=int(args.step_floor),
        #                          signal=args.public_signal)  # 假设bid 也从分布中取


def generate_action_rl(self, obs):
    # print('self', self)
    # print('obs', obs)
    # encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

    signal = obs['public_signal']

    action, _, _ = self.algorithm.compute_single_action(
        encoded_input_emb(signal, action_space_num=self.args.bidding_range)
    )

    # Decode the encoded action
    #action = encoded_action % (self.algorithm.bidding_range)

    self.record_action(action)
    return action
    # return 1


def update_policy_rl(self, obs, reward, done):
    #last_action = self.get_latest_action()
    # true_value = []
    #
    # if done:
    #     true_value = self.get_latest_true_value()
    # else:
    #     true_value = self.get_last_round_true_value()
    # encoded_action = last_action + true_value * (self.algorithm.bidding_range)
    # self.algorithm.update_policy(encoded_action, reward=reward)

    if obs['allocation']==1:
        #only record allocation =1
        self.record_true_value(obs['true_value'])
        self.record_signal(obs['public_signal'])
        self.record_reward(reward)

        self.record_data_num+=1


        if self.record_data_num % 32==0 : #self.algrithm.update_frequency:
            train_dataset = ray.data.from_items([{"x": self.signal_history[-x],
                                              "y": self.reward_history[-x]
                                              } for x in range(32)])

    # to be done:
        #update policy

    #


    return


def receive_observation_rl(self, args, budget, agent_name, agt_idx,
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

    self.record_obs(obs)

    return obs

def test_policy_rl(self,state):

    signal = state['public_signal']


    action, _, _ = self.algo.compute_single_action(
        encoded_input_emb(signal, action_space_num=self.args.bidding_range)
    )

    return action
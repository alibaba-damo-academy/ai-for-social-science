import random
import numpy as np

import sys

sys.path.append('../')
from env.multi_dynamic_env import *
from mini_env.open_auction import *
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
from pricing_signal.private_signal_generator import *

from test_mini_env.test_signal_game_example_function import *
from test_mini_env.rllib_algo_example import *
from test_mini_env.pytorch_algo_example import *

from copy import deepcopy
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
## mechanism setting
parser.add_argument('--mechanism', type=str, default='second_price',
                    help='mechanisim name')  # second_price | first_price | third_price | pay_by_submit | vcg
parser.add_argument('--allocation_mode', type=str, default='highest',
                    help='allocation mode')  # highest | vcg

parser.add_argument('--overbid', type=bool, default=True,
                    help='allow overbid or not')

parser.add_argument('--folder_name', type=str, default='open_auction_results',
                    help='store_figure path')
parser.add_argument('--print_key', type=bool, default=False,
                    help='whether to print procedure')
## agent setting
parser.add_argument('--bidding_range', type=int, default=20,
                    help='bidding range ')
parser.add_argument('--valuation_range', type=int, default=10,
                    help='valuation range ')
parser.add_argument('--assigned_valuation', type=int, default=None, action='append',
                    help='assign special valuation towards agent ')

parser.add_argument('--action_mode', type=str, default='same',  # same
                    help='defalut is the same mode, div means action is the true value div a number ')
parser.add_argument('--div_const', type=int, default=10,  # same
                    help='div mode parameter: compute the true bid by true value multiply the action / div_const')

## env basic setting
parser.add_argument('--player_num', type=int, default=4,
                    help='Set agent number')
parser.add_argument('--env_iters', type=int, default=10000,  # 5000000
                    help='env iteration number ')
parser.add_argument('--exploration_epoch', type=int, default=1000,  # 100001,
                    help='exploration epoch ')

##  print setting
parser.add_argument('--estimate_frequent', type=int, default=100,  # 4000  # 300003,
                    help='estimate_frequent')

##  experiment basic setting
parser.add_argument('--gpu', type=str, default='2,3',
                    help='Set CUDA_VISIBLE_DEVICES')
parser.add_argument('--exp_id', type=int, default='3',
                    help='experiment id')
parser.add_argument('--seed', type=int, default=0,
                    help='seed for reproducibility')

## env revenue record setting
parser.add_argument('--revenue_record_start', type=int, default=0,  # 0001,
                    help='record mechanism revenue start epoch (default the same to the exploration epoch)')

parser.add_argument('--revenue_averaged_stamp', type=int, default=200,
                    help='revenue averaged on how many epochs ')

parser.add_argument('--record_efficiency', type=int, default=0,
                    help='record auction efficiency or not [0,1]  ')
#
parser.add_argument('--inner_cooperate', type=int, default=0,
                    help='allow inner inner_cooperate item or not [0,1]  ')
parser.add_argument('--inner_cooperate_id', type=int, default=None, action='append',
                    help='allow inner inner_cooperate id (exceed the agent number equals no)  ')
parser.add_argument('--cooperate_pay_limit', type=int, default=0,
                    help='allow pay exceed their limit or  no ')

parser.add_argument('--value_div', type=int, default=0,
                    help='div the winning valuation or not  ')

# reward function argument
parser.add_argument('--reward_shaping', type=str, default=None,
                    help='reward_shaping function [CRRA,CARA]')
parser.add_argument('--reward_shaping_param', type=float, default=None, action='append',
                    help='reward_shaping function parameters')

# budget setting
parser.add_argument('--budget_mode', type=str, default=None,
                    help='choose budget mode in [None, budget_with_punish]')
parser.add_argument('--budget_sampled_mode', type=int, default=0,  # int budget only support for now
                    help='choose budget sampled mode in [0,1],0 means determinstic and 1 means uniform sampled from the distribution')
parser.add_argument('--budget_param', type=float, default=None, action='append',
                    help='set each agent budget (the reset budget is set to the last budget)')
parser.add_argument('--budget_punish_param', type=float, default=None, action='append',
                    help='set each agent budget punish (the reset budget is set to the last budget)')

# communication
parser.add_argument('--communication', type=int, default=0,
                    help='allow communication or not [0,1]  ')
parser.add_argument('--communication_type', type=str, default=None,
                    help='allow communication or not [None,value,info]  ')
parser.add_argument('--cm_id', type=int, default=None, action='append',
                    help='allow communication agent id (exceed the agent number equals no)  ')
# signal
parser.add_argument('--public_signal', type=int, default=1,
                    help='allow public signal or not [0,1],   ')

parser.add_argument('--public_signal_dim', type=int, default=1,
                    help='dimension of the public signal dim')

parser.add_argument('--public_signal_range', type=int, default=5,
                    help='upper bound of the public signal of each dim  ')

parser.add_argument('--private_signal', type=int, default=0,
                    help='allow private signal or not [0,1],   ')

parser.add_argument('--private_signal_dim', type=int, default=1,
                    help='dimension of the private signal dim')

parser.add_argument('--private_signal_range', type=int, default=5,
                    help='upper bound of the private signal of each dim  ')

#
parser.add_argument('--agt_obs_public_signal_dim', type=int, default=2,  # later change to the append prosper
                    help='each agent observe which dim of the public signal')

parser.add_argument('--value_to_signal', type=int, default=0,  # later change to the append prosper
                    help='allow generate value first then signal or not [0,1]')

parser.add_argument('--value_generator_mode', type=str, default='sum',  # later change to the append prosper
                    help='signal to value function : using [sum|mean ..]')

parser.add_argument('--speicial_agt', type=str, default=None,  # later change to the append prosper
                    help='speical dealed agent name ')

parser.add_argument('--noisy_observation', type=int, default=0,
                    help='allow public signal or not [0,1]')

parser.add_argument('--public_signal_asym', type=int, default=0,
                    help='Asymmetric property of the public signal')
parser.add_argument('--public_bidding_range_asym', type=int, default=0,
                    help='Asymmetric bidding range for each bidder in the public value scenario')

parser.add_argument('--public_signal_lower_bound', type=int, default=0,
                    help='lower bound of the public signal')
parser.add_argument('--public_signal_spectrum', type=int, default=0,
                    help='signal spectrum experiment')

parser.add_argument('--round', type=int, default=1,
                    help='how much round to apply with mini env')

parser.add_argument('--step_floor', type=int, default=10000,
                    help='how much epoch to decay with explore')
parser.add_argument('--smart_step_floor', type=str, default=None,
                    help='smart_step_floor')
parser.add_argument('--extra_round_income', type=float, default=0.0,
                    help='extra_round_income per round')

parser.add_argument('--self_play', type=int, default=0,
                    help='open self play or not')
parser.add_argument('--self_play_id', type=int, default=None, action='append',
                    help='assign self_play agent id')

parser.add_argument('--multi_item_decay', type=float, default=1,
                    help='Whether we add reward decay for agent obtaining multiple items within [0,1]')

# Multi-item setting
parser.add_argument('--item_num', type=int, default=1, help="number of items to bid")

# algorithm
parser.add_argument('--algorithm', type=str, default='Multi_arm_bandit',
                    help='Multi-arm-bandit or other deep model algorithm [Multi_arm_bandit,Deep]')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='deep learning model learning rate ')

parser.add_argument('--batchsize', type=int, default=16,
                    help='the deep learning model update frequent (the stored buffer size) ')

parser.add_argument('--cfr_loss', type=int, default=0,
                    help='the deep learning model cfr loss or not (the stored buffer size) ')

parser.add_argument('--agt_independent_private_value_func', type=int, default=0,
                    help='agt has independent private value func  or not (0 means the same reflect function) ')

# env name
parser.add_argument('--env_name', type=str, default='normal',
                    help='the multi env name:[normal | signal_game]  ')
parser.add_argument('--test_mini_env', type=int, default=0,
                    help='test_mini_env or not  ')
# --open auction config
parser.add_argument('--max_bidding_times', type=int, default=100,
                    help='maximal raise bidding time in each opening auction  ')
parser.add_argument('--minimal_raise', type=int, default=1,
                    help='minimal raise in each opening auction  ')

supported_function = ['CRRA', 'CARA']  # supported_reward_shaping_function
rl_lib_algorithm = ['DQN']


def easy_signal_setting(args, mode='default'):
    args.env_iters = 100000  # 0

    args.estimate_frequent = 20000  # 100000

    args.revenue_averaged_stamp = 10000

    args.player_num = 4

    args.max_bidding_times = 100
    args.minimal_raise = 1

    if mode == 'default':
        pass
    elif mode == 'pytorch':
        # set to mode 1
        args.overbid = False
        args.public_signal_dim = 1  # 1 dimension public
        args.agt_obs_public_signal_dim = 1  # each agt observe 1 dim signal
        args.public_signal_range = 10  # signal range from [0-10] int
        args.bidding_range = 10

        args.value_generator_mode = 'sum'  # signal sum
        args.folder_name = 'open_auction_results'

        args.exploration_epoch = 20000  # 0

        args.step_floor = 10000

    elif mode == 'pytorch_v2':
        # set to mode 1
        args.overbid = False
        args.public_signal_dim = 1  # 1 dimension public
        args.agt_obs_public_signal_dim = 1  # each agt observe 1 dim signal
        args.public_signal_range = 100  # signal range from [0-10] int
        args.bidding_range = 100

        args.value_generator_mode = 'sum'  # signal sum
        args.folder_name = 'open_auction_results'
        args.exp_id = 5

        args.batchsize = 16
        args.cfr_loss = 1

        args.step_floor = 10000

        args.env_iters = 400 * 10000
        args.estimate_frequent = 40 * 10000  # 100000
        args.exploration_epoch = 100000  # 40000

        args.revenue_averaged_stamp = 10000




    elif 'mode1' in mode:
        # set to mode 1
        args.overbid = False
        args.public_signal_dim = 1  # 1 dimension public
        args.agt_obs_public_signal_dim = 1  # each agt observe 1 dim signal
        args.public_signal_range = 10  # signal range from [0-10] int
        args.bidding_range = 10

        args.value_generator_mode = 'sum'  # signal sum
        args.folder_name = 'open_auction_results'

        args.exploration_epoch = 20000  # 0

        args.env_iters = 400000  # 0

        args.estimate_frequent = 50000  # 100000

        args.step_floor = 50000

        args.mechanism = 'first_price'

        if 'private' in mode:
            args.private_signal=1
            args.private_signal_range=10
            args.exp_id=4



    elif mode == 'mode2':
        # set to mode 2
        args.overbid = False
        args.public_signal_dim = 1  # 1 dimension public
        args.agt_obs_public_signal_dim = 1  # each agt observe 1 dim signal
        args.public_signal_range = 20  # signal range from [0-10] int
        args.bidding_range = 10

        args.value_generator_mode = 'sum'  # signal sum
        args.folder_name = 'open_auction_results'

        args.exploration_epoch = 100000

        args.step_floor = 10000
    elif mode == 'mode3':
        # set to mode 3
        args.test_mini_env = 1
        args.overbid = False
        args.public_signal_dim = 1  # 1 dimension public
        args.agt_obs_public_signal_dim = 1  # each agt observe 1 dim signal
        args.public_signal_range = 10  # signal range from [0-10] int
        args.bidding_range = 10

        args.value_generator_mode = 'sum'  # signal sum
        args.folder_name = 'open_auction_results'

        args.exploration_epoch = 100000

        args.step_floor = 10000

    return args


def adjust_dynamic_env(dynamic_env, args, mode='default'):
    if mode == 'default':
        return
    elif mode == 'mode1':
        public_signal_generator = platform_Signal_v1(dtype='Discrete', feature_dim=args.public_signal_dim,
                                                     lower_bound=0, upper_bound=args.public_signal_range,
                                                     generation_method='uniform',
                                                     value_to_signal=args.value_to_signal,
                                                     public_signal_asym=args.public_signal_asym
                                                     )  # assume upperbound

        dynamic_env.set_public_signal_generator(public_signal_generator)

    elif mode == 'mode2':
        # signal 0-20
        # value =0-10
        dynamic_env.set_env_public_signal_to_value(
            platform_public_signal_to_value=platform_public_signal_to_value
        )
    elif mode == 'mode3':
        public_signal_generator = platform_Signal_v2_rnd(dtype='Discrete', feature_dim=args.public_signal_dim,
                                                         lower_bound=1, upper_bound=args.public_signal_range,
                                                         generation_method='uniform',
                                                         value_to_signal=args.value_to_signal,
                                                         public_signal_asym=args.public_signal_asym
                                                         )  # assume upperbound

        dynamic_env.set_public_signal_generator(public_signal_generator)
        # set s->v func
        dynamic_env.set_env_public_signal_to_value(
            platform_public_signal_to_value=platform_public_signal_to_value_mode3
        )
        # set the update policy
        dynamic_env.set_update_env_public_signal(update_env_public_signal=test_update_env_public_signal)

    elif mode == 'mode1_private_only':

        dynamic_env.set_public_signal_generator(None)

        private_signal_generator = base_private_Signal_generator(
            dtype='Discrete', default_feature_dim=args.private_signal_dim,
            default_lower_bound=0,
            default_upper_bound=args.private_signal_range,
            default_generation_method='uniform',
            default_value_to_signal=0,
            agent_name_list=dynamic_env.agent_name_list
        )
        private_signal_generator.generate_default_iid_signal()

        dynamic_env.set_private_signal_generator(private_signal_generator)


mode ='mode1' #  'mode1_private_only'
# 'mode1' #   'mode1_private_only' #'pytorch_v2'
rl_lib_available = False
pytorch_lib_avalable = False  # True
only_winner_know_true_value = False


def main(args):
    set_seed(seed=args.seed)

    # Initialize the auction allocation/payment policy of the auctioneer

    # easy mode change
    args = easy_signal_setting(args, mode=mode)
    print(args)

    dynamic_env = open_ascending_auction(round=args.round, item_num=args.item_num, mini_env_iters=10000,
                                         agent_num=args.player_num, args=args,
                                         policy=Policy(allocation_mode=args.allocation_mode,
                                                       payment_mode=args.mechanism,
                                                       action_mode=args.action_mode,
                                                       args=args),
                                         set_env=None,
                                         init_from_args=True,  # init from args rather than input params ->
                                         private_signal_generator=None,
                                         signal_type='normal', winner_only=only_winner_know_true_value,
                                         max_bidding_times=args.max_bidding_times, minimal_raise=args.minimal_raise

                                         )

    env = dynamic_env.get_mini_env()
    print(env.agents)

    while dynamic_env.get_current_round() < dynamic_env.get_round():  # only play once

        # clean the old reward list
        dynamic_env.init_reward_list()

        # #set agent
        # dynamic_env.init_global_rl_agent()  #now we assign agent by hand-design (see agent_generate)
        # # [Implementation idea]: first init the default list of all agents,
        # # then replace the submitted agents into the default list

        # self_custom
        # new_eric_agent=eric_agent
        # dynamic_env.set_rl_agent(new_eric_agent)

        # method 1.
        # default_agent = greedy_simple_agent

        # adjust agent policy (include three metioned method)：
        if (rl_lib_available):
            custom_agent_name = get_custom_agent_name()
            custom_rl_agent = get_custom_rl_agent(args)

            # dynamic_env.set_rl_agent(rl_agent=custom_rl_agent,
            #                      agent_name=custom_agent_name) #adjust agent type

            # can provide more agent for user design based on the basic agent
            # example: add fixed agent (we provided example)
            # example: add greedy agent ( now different agents are designed due to different format obs)

        ## only adjust agent algorithm
        if (rl_lib_available):
            custom_algorithm = get_custom_algorithm_in_rllib(args)

            dynamic_env.set_rl_agent_algorithm(algorithm=custom_algorithm,
                                               agent_name=custom_agent_name)  # search agt.generate_action(obs)
            dynamic_env.set_rl_agent_generate_action(generate_action=generate_action_rl,
                                                     agent_name=custom_agent_name)
            dynamic_env.set_rl_agent_update_policy(update_policy=update_policy_rl,
                                                   agent_name=custom_agent_name)  # add function search agt.update_policy | user can user other infos in basic agents(see in  normal_agent.py )
            dynamic_env.set_rl_agent_receive_observation(receive_observation=receive_observation_rl,
                                                         agent_name=custom_agent_name)  # add function init_reward_list | search receive_observation()

            dynamic_env.set_rl_agent_test_policy(test_policy=test_policy_rl, agent_name=custom_agent_name
                                                 )
        # obs need more works to make a standard .

        if pytorch_lib_avalable:
            custom_agent_name = get_custom_agent_name()
            custom_algorithm = get_custom_algorithm_in_pytorch(args)

            def set_pytorch_for_agent(custom_agent_name, dynamic_env, custom_algorithm):
                dynamic_env.set_rl_agent_algorithm(algorithm=custom_algorithm,
                                                   agent_name=custom_agent_name)  # search agt.generate_action(obs)
                dynamic_env.set_rl_agent_generate_action(generate_action=generate_action_pytorch,
                                                         agent_name=custom_agent_name)
                dynamic_env.set_rl_agent_update_policy(update_policy=update_policy_pytorch,
                                                       agent_name=custom_agent_name)  # add function search agt.update_policy | user can user other infos in basic agents(see in  normal_agent.py )
                dynamic_env.set_rl_agent_receive_observation(receive_observation=receive_observation_pytorch,
                                                             agent_name=custom_agent_name)  # add function init_reward_list | search receive_observation()
                dynamic_env.set_rl_agent_test_policy(test_policy=test_policy_pytorch, agent_name=custom_agent_name
                                                     )

            # SET FOR ALL AGENT
            set_pytorch_for_agent(custom_agent_name='player_0', dynamic_env=dynamic_env,
                                  custom_algorithm=get_custom_algorithm_in_pytorch(args, seed=0))
            set_pytorch_for_agent(custom_agent_name='player_1', dynamic_env=dynamic_env,
                                  custom_algorithm=get_custom_algorithm_in_pytorch(args, seed=1))
            set_pytorch_for_agent(custom_agent_name='player_2', dynamic_env=dynamic_env,
                                  custom_algorithm=get_custom_algorithm_in_pytorch(args, seed=2))
            set_pytorch_for_agent(custom_agent_name='player_3', dynamic_env=dynamic_env,
                                  custom_algorithm=get_custom_algorithm_in_pytorch(args, seed=3))

        adjust_dynamic_env(dynamic_env, args=args, mode=mode)

        # next step
        dynamic_env.step()  # run the mini env and update policy each

        # rebuild mini env
        dynamic_env.reset_env()  # reinit


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.agt_obs_public_signal_dim > args.public_signal_dim:
        args.agt_obs_public_signal_dim = args.public_signal_dim

    main(args)

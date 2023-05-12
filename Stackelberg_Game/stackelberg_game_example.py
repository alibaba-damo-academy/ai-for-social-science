import random
import numpy as np

import sys

sys.path.append('../')
from env.multi_dynamic_env import *
from mini_env.base_signal_mini_env import *
from mini_env.simple_mini_env1 import *
from mini_env.stackelberg_env import *

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
from env.payment_rule import *
from env.policy import *
from agent_utils.action_encoding import *
from agent_utils.action_generation import *
from agent_utils.agent_updating import *
from agent_utils.obs_utils import *
from agent_utils.signal import *
from agent_utils.valuation_generator import *

from agent.agent_generate import *

from pricing_signal.plat_deliver_signal import *
from test_mini_env.test_signal_game_example_function import *
from test_mini_env.rllib_algo_example import *
from test_mini_env.pytorch_algo_example import *

from mechanism_design.deep_mechanism import *
from mechanism_design.deep_allocation import *
from mechanism_design.deep_payment import *

from stackelberg_agent_utils import *
from stackelberg_game_mechanism import *
from stackelberg_env_utils import *

import test_player_class
import test_player_func

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

parser.add_argument('--overbid', type=bool, default=False,
                    help='allow overbid or not')
##  print setting
parser.add_argument('--estimate_frequent', type=int, default=400002,  # 300003,
                    help='estimate_frequent')

parser.add_argument('--folder_name', type=str, default='tmp',
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
parser.add_argument('--env_iters', type=int, default=5000000,  # 5000000
                    help='env iteration number ')
parser.add_argument('--exploration_epoch', type=int, default=400000,  # 100001,
                    help='exploration epoch ')

##  experiment basic setting
parser.add_argument('--gpu', type=str, default='1,2,3',
                    help='Set CUDA_VISIBLE_DEVICES')
parser.add_argument('--exp_id', type=int, default='2',
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
parser.add_argument('--public_signal', type=int, default=0,
                    help='allow public signal or not [0,1],   ')

parser.add_argument('--public_signal_dim', type=int, default=1,
                    help='dimension of the public signal dim')

parser.add_argument('--public_signal_range', type=int, default=5,
                    help='upper bound of the public signal of each dim  ')

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
                    help='deep learning model batchsize ')

parser.add_argument('--update_frequent', type=int, default=200,
                    help='the deep learning model update frequent (the stored buffer size) ')

# env name
parser.add_argument('--env_name', type=str, default='normal',
                    help='the multi env name:[normal | signal_game]  ')

parser.add_argument('--test_player', type=str, default='class',
                    choices=['class', 'func'])
# discount
parser.add_argument('--discount_range', type=int, default=10, help='discount range: 1-discount_range')

# information
parser.add_argument('--info_signal_range', type=int, default=10, help='informaiton signal range')
parser.add_argument('--info_algo_range', type=int, default=9, help='informaiton algorithm output range')

# Stackelberg game
parser.add_argument('--stackelberg_epoch', type=int, default=10, help='trainning rounds of stackelberg game ')

supported_function = ['CRRA', 'CARA']  # supported_reward_shaping_function
rl_lib_algorithm = ['DQN']

replace_agent = False


def easy_signal_setting(args, mode='default'):
    args.mechanism = 'customed'
    args.player_num = 3

    args.overbid = True  # overbid: bid can exceed value(sell price >= cost)
    args.bidding_range = 20  # bid range [0,20)
    args.valuation_range = 10  # valuation range is cost range here [0,10)
    args.allocation_mode = 'lowest'  # allocate to bidder with the lowest price

    args.discount_range = 10 - 1  # discount [1,10)
    # There are two type of information:
    # info_signal = 0 => no information is revealed
    # info_signal: [1,10) => player 0 reveal information (such as truthfully reveal discount)
    args.info_signal_range = args.discount_range + 1  # info signal range [0,10)
    args.info_algo_range = args.discount_range  # info algo range is revealed information -> [1,10)

    args.exp_id = 0  # experiment id (for result saving)
    args.folder_name = 'stackelberg'  # result folder name

    args.stackelberg_epoch = 4 * 5  # stackelberg game rounds
    args.env_iters = 80 * 10000  # environment iterations
    args.estimate_frequent = 40 * 10000  # estimate per K iters
    args.revenue_averaged_stamp = 10000
    args.exploration_epoch = 20 * 10000
    args.step_floor = 4 * 10000

    args.round = 10

    return args


def main(args):
    # set seed
    np.random.seed(42)
    random.seed(42)

    args = easy_signal_setting(args)    # set args
    print(args)

    # set_seed(seed = args.seed)

    dynamic_env = stackelberg_game(round=args.round, item_num=args.item_num, mini_env_iters=10000,
                                   agent_num=args.player_num, args=args,
                                   policy=stackelberg_Policy(allocation_mode=args.allocation_mode,
                                                             payment_mode='lowest',
                                                             action_mode=args.action_mode,
                                                             args=args),
                                   set_env=None,
                                   init_from_args=True,  # init from args rather than input params ->
                                   winner_only=False,
                                   leading_agent_id=0,
                                   reward_mode='revenue',
                                   leading_agent_update_flag=True

                                   )
    dynamic_env.init_reward_list()


    env = dynamic_env.get_mini_env()
    print(env.agents)
    leading_agent_name = dynamic_env.leading_agent_name

    #####################################################################################
    # In a market where 3 suppliers compete with each other, selling identical products,
    # everyone receives his/her cost and bid with a price at each round,
    # who bids the lowest price will win the opportunity to sell his/her products at the
    # price he/her bids.
    #
    # (Step1)
    # ROUND 0: every player bids in order to maximize its revenue
    #
    # (Step2)
    # Then player 0(leader) decides to reduce price to seize the market(maximize income):
    # ROUND 1: other players still use its policy in round 0, with player 0 reduce price.
    # ROUND 2: other players response to maximize its revenue when player 0 reduce price.
    # ROUND 3: player 0 reveal a signal(truthfully) about his discount,
    # and other players response to the signal and its cost to maximize its recenue.
    #
    # (Step3)
    # After that, we consider a Stackelberg game:
    # (1) P0 choose discount and information, others fixed policy
    # (2) P0 fix policy, others see information and bid
    # step 1,2 will continue to cycle.
    #####################################################################################
    # round 0
    # everyone update bid policy, no discount, no information
    dynamic_env.turnon_follower_agent_update()
    dynamic_env.turnon_leading_agent_update()
    dynamic_env.disable_discount()
    dynamic_env.disable_information()

    platform_reward_r0 = dynamic_env.step()  # run the mini env and update policy

    # export agent algo
    trained_agents = dynamic_env.export_all_agent()

    # round 1  decrease + others not update

    # leader/follower bid policy no update, leader discount policy update
    # enable discount, no information
    dynamic_env.re_init()
    dynamic_env.turnoff_leading_agent_update()
    dynamic_env.turnoff_follower_agent_update()
    dynamic_env.enable_discount()
    dynamic_env.turnon_leader_discount_update()
    dynamic_env.disable_information()
    dynamic_env.turnoff_leader_info_update()
    # set reward mode for leader
    dynamic_env.adjust_reward_mode(reward_mode='income')

    # set agents
    # set leader
    leader_agt = deepcopy(trained_agents[leading_agent_name])
    leader_agt = set_leader_algorithm(leader_agt, args)
    dynamic_env.set_rl_agent(rl_agent=leader_agt, agent_name=leading_agent_name)
    # set folllowers
    for agent_name, agt in trained_agents.items():
        if agent_name != leading_agent_name:
            dynamic_env.set_rl_agent(rl_agent=deepcopy(agt), agent_name=agent_name)

    # run env
    platform_reward_r1 = dynamic_env.step()  # run the mini env and update policy

    # export agent algo
    trained_agent_list_r1 = dynamic_env.export_all_agent()

    ########

    # round 2 --- follower update

    # leader bid policy no update, leader discount policy no update
    # follower bid policy update
    # enable discount, no information
    dynamic_env.re_init()
    dynamic_env.turnoff_leading_agent_update()
    dynamic_env.turnon_follower_agent_update()
    dynamic_env.enable_discount()
    dynamic_env.turnoff_leader_discount_update()
    dynamic_env.disable_information()
    dynamic_env.turnoff_leader_info_update()
    # set reward mode for leader
    dynamic_env.adjust_reward_mode(reward_mode='income')

    # set agents
    # set leader
    leader_agt = deepcopy(trained_agent_list_r1[leading_agent_name])
    dynamic_env.set_rl_agent(rl_agent=leader_agt, agent_name=leading_agent_name)
    # set folllowers
    for agent_name, agt in trained_agents.items():
        if agent_name != leading_agent_name:
            new_agent = deepcopy(agt)
            # reset algorithm for better training (optional)
            new_agent.reset_algo()
            dynamic_env.set_rl_agent(rl_agent=new_agent, agent_name=agent_name)

    # run env
    platform_reward_r2 = dynamic_env.step()  # run the mini env and update policy

    # export agent algo
    trained_agent_list_r2 = dynamic_env.export_all_agent()

    # round 3 --- information reveal

    # leader bid/discount policy no update, leader info policy no update(truthful)
    # follower bid policy update
    # enable discount, enable info
    dynamic_env.re_init()
    dynamic_env.turnoff_leading_agent_update()
    dynamic_env.turnon_follower_agent_update()
    dynamic_env.enable_discount()
    dynamic_env.turnoff_leader_discount_update()
    dynamic_env.enable_information()
    dynamic_env.turnoff_leader_info_update()
    dynamic_env.init_truthful_info()  # initialize information (set truthfully reveal)
    # set reward mode for leader
    dynamic_env.adjust_reward_mode(reward_mode='income')

    # set agents
    # set leader
    leader_agt = deepcopy(trained_agent_list_r1[leading_agent_name])
    dynamic_env.set_rl_agent(rl_agent=leader_agt, agent_name=leading_agent_name)
    # set folllowers
    for agent_name, agt in trained_agents.items():
        if agent_name != leading_agent_name:
            new_agent = deepcopy(agt)
            # reset algorithm for better training (optional)
            new_agent.reset_algo()
            dynamic_env.set_rl_agent(rl_agent=new_agent, agent_name=agent_name)

    # run env
    platform_reward_r3 = dynamic_env.step()  # run the mini env and update policy each
    # export agent algo
    trained_agent_list_r3 = dynamic_env.export_all_agent()

    ####################
    # Stackelberg Game #
    ####################
    # records
    stkbg_agents_revenues = []
    stkbg_agents_incomes = []
    stkbg_platform_rewards = []
    # init agents & env
    init_stkbg_agents = init_agents_for_stkbg(trained_agents, leading_agent_name, args)
    dynamic_env.init_truthful_info()  # initialize information (set truthfully reveal)
    # run stackelberg game
    cur_agents = init_stkbg_agents
    for epoch in range(args.stackelberg_epoch):
        print('Stackelberg Game Epoch:', epoch)
        # re-init env
        reinit_env_for_stkbg(dynamic_env, cur_agents)
        # set env update mode
        set_env_update_mode(dynamic_env, epoch)
        # run env
        platform_reward = dynamic_env.step()
        # record results
        stkbg_platform_rewards.append(platform_reward)
        stkbg_agents_revenues.append(get_agents_revenues(cur_agents, args))
        stkbg_agents_incomes.append(get_agents_incomes(cur_agents, args))
        # plot results
        print('plotting stackelberg results...')
        plot_trained_agents_stkbg(cur_agents, leading_agent_name, args, epoch)

    ####################

    # plot all results trained in stkbg game
    print('plotting all results in stackelberg game...')
    plot_all_results_stkbg(stkbg_platform_rewards,stkbg_agents_revenues,stkbg_agents_incomes,args,leading_agent_name)

    # print rewards
    print('---------------------------------------------')
    print('All rewards in example rounds:')
    trained_agents_list_all = [trained_agents, trained_agent_list_r1, trained_agent_list_r2, trained_agent_list_r3]
    platform_reward_all = [platform_reward_r0, platform_reward_r1, platform_reward_r2, platform_reward_r3]
    for i, agents in enumerate(trained_agents_list_all):
        print('round', i)
        print('sell price:\t', - platform_reward_all[i])
        for agent_name, agt in agents.items():
            print(agent_name,
                  '\trevenue:',
                  sum(agt.sp_revenue_record[-args.revenue_averaged_stamp:]) / args.revenue_averaged_stamp,
                  '\tincome:',
                  sum(agt.sp_income_record[-args.revenue_averaged_stamp:]) / args.revenue_averaged_stamp
                  )

    # print stkbg rewards
    print('---------------------------------------------')
    print('All rewards in Stackelberg game:')
    all_agent_name = trained_agents.keys()
    for epoch in range(args.stackelberg_epoch):
        if epoch % 2 == 0:
            if epoch % 4 == 0:
                postfix = 'update discount'
            else:
                postfix = 'update info'
        else:
            postfix = 'follow update'
        print('Stackelberg Game Epoch:', epoch, postfix)
        print('sell price:\t', -stkbg_platform_rewards[epoch])
        for agent_name in all_agent_name:
            print(agent_name, '\trevenue:', stkbg_agents_revenues[epoch][agent_name],
                  '\tincome:', stkbg_agents_incomes[epoch][agent_name])


def replace_agent_func(dynamic_env):
    # #set agent
    # dynamic_env.init_global_rl_agent()  #now we assign agent by hand-design (see agent_generate)
    # # [Implementation idea]: first init the default list of all agents,
    # # then replace the submitted agents into the default list

    if args.test_player == 'class':
        user_player = test_player_class.user_player_class(args=args,
                                                          agent_name='player_3',
                                                          highest_value=args.valuation_range - 1)
        custom_agent_name = user_player.get_custom_agent_name()

        dynamic_env.set_rl_agent(rl_agent=user_player,
                                 agent_name=custom_agent_name)  # adjust agent type
        dynamic_env.assign_generate_true_value(agent_name_list=[custom_agent_name])

    elif args.test_player == 'func':
        custom_agent_name = test_player_func.get_custom_agent_name()
        dynamic_env.set_rl_agent_generate_action(generate_action=test_player_func.generate_action,
                                                 agent_name=custom_agent_name)
        dynamic_env.set_rl_agent_update_policy(update_policy=test_player_func.update_policy,
                                               agent_name=custom_agent_name)  # add function search agt.update_policy | user can user other infos in basic agents(see in  normal_agent.py )
        dynamic_env.set_rl_agent_receive_observation(receive_observation=test_player_func.receive_observation,
                                                     agent_name=custom_agent_name)  # add function init_reward_list | search receive_observation()
    else:
        raise NotImplementedError(f'test_player mode = {args.test_player} not implemented!')

    # adjust agent policy (include three metioned method)：
    #         # can provide more agent for user design based on the basic agent
    #         # example: add fixed agent (we provided example)
    #         # example: add greedy agent ( now different agents are designed due to different format obs)

    # ## only adjust agent algorithm
    # dynamic_env.set_rl_agent_algorithm(algorithm=custom_algorithm,
    #                                    agent_name=custom_agent_name) #search agt.generate_action(obs)
    # obs need more works to make a standard .

    return


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.agt_obs_public_signal_dim > args.public_signal_dim:
        args.agt_obs_public_signal_dim = args.public_signal_dim

    main(args)

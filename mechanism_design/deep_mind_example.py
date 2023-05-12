import random
import numpy as np


import sys
sys.path.append('../')
import logging
from env.multi_dynamic_env import *
from mini_env.base_signal_mini_env import *
from mini_env.simple_mini_env1 import *
from mini_env.mechanism_opt_example_env import *


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
                    help='allocation mode') # highest | vcg

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
parser.add_argument('--seed', type=int, default = 0,
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

parser.add_argument('--agt_obs_public_signal_dim', type=int, default=2, #later change to the append prosper
                    help='each agent observe which dim of the public signal')

parser.add_argument('--value_to_signal', type=int, default=0, #later change to the append prosper
                    help='allow generate value first then signal or not [0,1]')

parser.add_argument('--value_generator_mode', type=str, default='sum', #later change to the append prosper
                    help='signal to value function : using [sum|mean ..]')

parser.add_argument('--speicial_agt', type=str, default=None, #later change to the append prosper
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

parser.add_argument('--multi_item_decay', type=float, default= 1,
                    help='Whether we add reward decay for agent obtaining multiple items within [0,1]')


# Multi-item setting
parser.add_argument('--item_num', type=int, default = 1, help="number of items to bid")


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

supported_function = ['CRRA', 'CARA']  # supported_reward_shaping_function
rl_lib_algorithm = ['DQN']

# all-pay
parser.add_argument('--weights', type=float, default=0.5, 
                    help='weights of all pay')

replace_agent=False

def easy_signal_setting(args, mode='default'):
    args.env_iters = 2000000  # 0

    args.estimate_frequent = 100000  # 100000

    args.revenue_averaged_stamp = 50000

    args.exploration_epoch = 400000  # 0

    args.step_floor = 10000

    args.player_num = 3

    args.mechanism='deep'

    args.folder_name='mec_opt_40w_200w'
    args.bidding_range = 10
    args.valuation_range=10

    args.batchsize=64

    args.round=10

    args
    # args.max_bidding_times = 100
    # args.minimal_raise = 1


    return args


def setup_logger(args):
    level = getattr(logging, 'INFO', None)
    folder_path = os.path.join('results', args.folder_name, str(args.exp_id))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(folder_path, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    print(handler2.baseFilename)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)


def main(args):


    args=easy_signal_setting(args)
    setup_logger(args)
    logging.info(str(args))


    #set_seed(seed = args.seed)

    weights = args.weights
    logging.info(f'weights = {args.weights}')

    # Initialize the auction allocation/payment policy of the auctioneer
    deep_mechanism = deep_policy2(
        allocation_mode=args.allocation_mode,
        payment_mode='customed',
        action_mode=args.action_mode,
        weights=args.weights,
        args=args
    )
    # build deep payment



    dynamic_env = opt_examle_env(round=args.round, item_num=args.item_num, mini_env_iters=10000,
                                   agent_num=args.player_num, args=args,
                                   policy=deep_mechanism,
                                   set_env=None,
                                   init_from_args=True, #init from args rather than input params ->
                                    winner_only=False,
                                 fixed_truthful=False, #apply truthful agent or not

        )

    env = dynamic_env.get_mini_env()
    print(env.agents)


    dataset=[]

    import torch

    while dynamic_env.get_current_round() < dynamic_env.get_round(): #only play once


        # clean the old reward list
        dynamic_env.init_reward_list()


        if replace_agent:
            replace_agent_func()


        #next step
        platform_reward = dynamic_env.step() #run the mini env and update policy each

        logging.info(platform_reward)

        dataset.append(torch.tensor([weights,platform_reward]))

        # #
        # dataset = dynamic_env.generate_dataset(data_num=args.batchsize)
        # #print(dataset)
        #
        #
        #
        # # train the mechanism
        # deep_payment_rule.model_train(dataset=dataset,agent_name_list=dynamic_env.agent_name_list,batchsize=args.batchsize)
        #
        # #
        #

        # rebuild mini env
        dynamic_env.re_init() #reinit
        #deep_mechanism.close_grad()


    #print_loss
    #deep_payment_rule.print_loss()







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

def customed_payment(weights,bids,possible_agents,winner):

    # print('bids', bids)
    # print('winner', winner)
    # print('possible_agents', possible_agents)
    payment={agent: 0 for agent in possible_agents}

    second_highest_bid = get_rank(bids,possible_agents,rank=2)

    for i in possible_agents:

        if i == winner:  # in winner_list: #== winner :

            payment[i] = -weights * bids[i]
        elif bids[i]==second_highest_bid: #second high
            payment[i] = -bids[i] * (1-weights)
        else:
            payment[i] = 0


    return payment

class deep_policy2(Policy):

    def __init__(self,allocation_mode='deep', payment_mode='deep',action_mode='same',weights=0.5,args=None):
        super(deep_policy2).__init__()

        self.possible_agents = None
        self.state = None
        self.allocation_mode = allocation_mode
        self.payment_mode = payment_mode
        self.action_mode = action_mode
        self.args = args
        #
        self.weights=weights


        self.last_winner = None
        self.last_payment = None

        self.allocation_net = None
        self.payment_net = None

    def compute_allocation(self, state,reserve_price=0):
        if self.action_mode=='div':
            state = de_transform_state(state=state, args=self.args)
        if self.allocation_mode == 'highest':
            winner = highest_allocation(state=state, possible_agents=self.possible_agents,reserve_price=reserve_price)

        elif self.allocation_mode == 'deep':
            winner = self.allocation_net.compute_allocation(state=state, possible_agents=self.possible_agents,reserve_price=reserve_price)


        self.last_winner=winner


        return winner

    def compute_payment(self, state, winner=None):

        if self.action_mode == 'div':
            state = de_transform_state(state=state, args=self.args)

        if self.payment_mode == 'second_price':
            payment = second_price_rule(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'first_price':
            payment = first_price_rule(state, possible_agents=self.possible_agents, winner=winner)

        elif self.payment_mode == 'third_price':
            payment = third_price_rule(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'pay_by_submit':
            payment = pay_by_submit(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'deep':
            # add deep net payment
            payment = self.payment_net.compute_payment(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode =='customed':
            # customed payment with weight

            payment = customed_payment(weights=self.weights,bids=state, possible_agents=self.possible_agents, winner=winner)
        else:
            logging.info('not find the payment mode' + str(self.payment_mode))

        self.last_payment = payment

        return payment

    def assign_payment_net(self,payment_net):
        self.payment_net=payment_net

    def assign_allocation_net(self,allocation_net):
        self.allocation_net = allocation_net

    def export_payment_net(self):
        return deepcopy(self.payment_net)

    def export_allocation_net(self):
        return deepcopy(self.allocation_net)


    def close_grad(self):
        if self.payment_mode== 'deep' and self.payment_net is not None:
            self.payment_net.close_grad()

        if self.allocation_mode== 'deep' and self.allocation_net is not None:
            self.allocation_net.close_grad()


    def open_grad(self):

            if self.payment_mode == 'deep' and self.payment_net is not None:
                self.payment_net.open_grad()

            if self.allocation_mode == 'deep' and self.allocation_net is not None:
                self.allocation_net.open_grad()


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.agt_obs_public_signal_dim >args.public_signal_dim:
        args.agt_obs_public_signal_dim = args.public_signal_dim




    main(args)

import random
import numpy as np
from env.customed_env_auction import *
from agent.normal_agent import *
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

from copy import deepcopy
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
## mechanism setting
parser.add_argument('--mechanism', type=str, default='second_price',
                    help='mechanisim name')  # second_price | first_price | third_price | pay_by_submit |

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
parser.add_argument('--revenue_record_start', type=int, default=100000,  # 0001,
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

# Multi-item setting
parser.add_argument('--item_num', type=int, default = 1, help="number of items to bid")




supported_function = ['CRRA', 'CARA']  # supported_reward_shaping_function
rl_lib_algorithm = ['DQN']


def main(args):

    set_seed(seed = args.seed)
    
    # Initialize the auction allocation/payment policy of the auctioneer
    test_policy = Policy(allocation_mode='highest', payment_mode=args.mechanism,\
                        action_mode=args.action_mode, args=args)

    # Initialize the simulation environment (e.g. three player, second price auction)
    env = customed_env(args, policy=test_policy)
    env.reset()
    print(env.agents)

    # print(args.inner_cooperate_id)

    test_policy.assign_agent(env.agents)
    agt_list = {agent: None for agent in env.agents}

    # set budget
    budget = None
    public_signal_generator=None
    public_signal_to_value=None

    if args.budget_mode is not None:
        budget = build_budget(args, agent_name_list=env.agents)

    if args.public_signal :
        if 'kl' in args.folder_name:
            public_signal_generator = KL_Signal(dtype='Discrete', feature_dim=args.public_signal_dim,
                                         lower_bound=0, upper_bound=args.public_signal_range,
                                         generation_method='uniform',
                                         value_to_signal=args.value_to_signal,
                                         value_generator_mode=args.value_generator_mode
                                         ) # assume upperbound
        else:
            public_signal_generator = Signal(dtype='Discrete', feature_dim=args.public_signal_dim,
                                         lower_bound=0, upper_bound=args.public_signal_range,
                                         generation_method='uniform',
                                         value_to_signal=args.value_to_signal, 
                                         public_signal_asym=args.public_signal_asym
                                         ) # assume upperbound
        if args.value_to_signal:
            value_generator_mode='fixed' 
        else:
            value_generator_mode = args.value_generator_mode
         
        public_signal_to_value = public_value_generator(signal_generator=public_signal_generator,
                                                        value_generator_mode=value_generator_mode,
                                                        valuation_range = args.valuation_range
                                                        )
        if args.public_signal_spectrum:  
            # Set up an interpolation experiment between pure private value and 
            # pure public value model
            new_public_signal_generator = deepcopy(public_signal_generator)
            new_public_signal_generator.lower_bound = args.public_signal_lower_bound
            new_public_signal_generator.upper_bound = args.public_signal_range - args.public_signal_lower_bound
            public_signal_to_value = public_value_generator(signal_generator=new_public_signal_generator,
                                                        value_generator_mode=value_generator_mode,
                                                        valuation_range = args.valuation_range
                                                        )

    # Initialize agent profile including inherent constraints (budget, bidding 
    # range ...), Bidding algorithm, interaction with other agents/env. 
    agt_list = generate_agent(env=env, args=args, agt_list=agt_list, budget=budget)

    epoch = 0
    extra_info = None
    public_signal=None

    # Record the simulation results
    revenue_record = env_record(record_start_epoch=args.revenue_record_start,  # start record env revenue number
                                averaged_stamp=args.revenue_averaged_stamp)
    
    for agent_name in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()
        # env.render()  # this visualizes a single game
        _obs = observation['observation']

        agt = agt_list[agent_name]
        agt_idx = env.agents.index(agent_name)

        # generate action for all agent
        if agent_name == env.agents[0] and (not truncation) and (not termination):  
            # start of a bidding episode since we loop according to the order of 
            # agent.

            #generate public signal
            if public_signal_generator is not None:
                last_public_value = public_signal_to_value.get_last_public_gnd_value()
                    #public_signal_to_value.generate_value(obs=None, gnd=True, weighted=False) # get value before update the new signal

                public_signal_generator.generate_signal(data_type='int')  #update the signal
                # for debug
                global_signal = public_signal_generator.get_whole_signal_realization()

            # Update auction info (signal realization) at the start of an auction
            update_agent_profile(env, args, agt_list, agent_name,
                                 budget,
                                 public_signal_generator=public_signal_generator,
                                 public_signal_to_value=public_signal_to_value
                                 )  # should update all the latest true value function
            # for debug
            global_signal = public_signal_generator.get_whole_signal_realization()

            extra_info = communication(env, args, agt_list, agent_name)

        if (args.action_mode == 'div' and _obs == args.bidding_range * args.valuation_range + 1) or \
                (args.action_mode == 'same' and _obs == args.bidding_range + 1):  
            # the first round not compute reward
            # In real scenario, signal realization/auction setup happens at the
            # start of a auction, and auction allocation happens at the end. We 
            # simply the implementation by setting both action to the start by
            # skipping them at the first round.
            print('first round')

        else: # Compute reward and update agent policy each round 
            allocation = info['allocation']
            # get true value for computing the reward
            if args.public_signal:
                true_value =last_public_value
            else:
                true_value = agt.get_last_round_true_value()

            # adjust each agent with his reward function
            final_reward = compute_reward(true_value=true_value, pay=reward,
                                          allocation=allocation,
                                          reward_shaping=check_support_reward_shaping(supported_function,
                                                                                      args.reward_shaping),
                                          reward_function_config=build_reward_function_config(args),
                                          user_id=agent_name, budget=budget,
                                          args=args, info=info  # compute based on the former budget
                                          )  # reward shaping

            last_action = agt.get_latest_action()
            last_state = []  # [last_true_value, last_action]  # [1-H, 0-H-1]
            # update poicy
            agt.update_policy(state=last_state, reward=final_reward)
            # record each step
            record_step(args, agt, allocation, agent_name, epoch, env, reward, revenue_record)

        if args.public_signal:
            # Given the public signal realization, generate the public signal
            # observable to the agent
            public_signal=public_signal_generator.get_partial_signal_realization(observed_dim_list=agt.get_public_signal_dim_list())
            agt.receive_partial_obs_true_value(true_value=public_signal_to_value.generate_value(obs=public_signal))

        obs = receive_observation(args, budget, agent_name, agt_idx, extra_info, observation,
                    public_signal = public_signal
                                  )

        # new round behavior
        # agt.generate_true_value()

        next_true_value = agt.get_latest_true_value()
        new_action = action_generation(args, agt, obs)  # get the next round action based on the observed budget

        ## cooperate infomation processing [eric]
        submit_info_to_env(args, agt_idx, test_policy, agent_name, next_true_value)

        if termination or truncation:
            env.step(None)
        else:
            env.step(new_action)
        ## print setting
        epoch += 1
        print_process(env, args, agt_list, new_action, agent_name, next_true_value, epoch, revenue_record)

        # env.render()  # this visualizes a single game


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.agt_obs_public_signal_dim >args.public_signal_dim:
        args.agt_obs_public_signal_dim = args.public_signal_dim


    print(args)

    main(args)

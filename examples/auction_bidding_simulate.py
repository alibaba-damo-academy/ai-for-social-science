import random
import numpy as np
from collections import defaultdict
import dill
from env.customed_env_auction import *
from agent.normal_agent import *
from rl_utils.solver import *
from rl_utils.reward_shaping import *
from plot.plot_util import *
from env.env_record import *
from env.Policy import *

from copy import deepcopy
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
## mechanism setting
parser.add_argument('--mechanism', type=str, default='second_price',
                    help='mechanisim name')

parser.add_argument('--overbid', type=bool, default=False,
                    help='allow overbid or not')
##  print setting
parser.add_argument('--estimate_frequent', type=int, default=300003,#100000,
                    help='estimate_frequent')

parser.add_argument('--folder_name', type=str, default='ucb',
                    help='store_figure path')
parser.add_argument('--print_key', type=bool, default=False,
                    help='whether to print procedure')
## agent setting
parser.add_argument('--bidding_range', type=int, default=10,
                    help='bidding range ')
parser.add_argument('--valuation_range', type=int, default=10,
                    help='valuation range ')
## env basic setting
parser.add_argument('--player_num', type=int, default=3,
                    help='Set agent number')
parser.add_argument('--env_iters', type=int, default=5000000,
                    help='env iteration number ')
parser.add_argument('--exploration_epoch', type=int, default=100000,#100001,
                    help='exploration epoch ')

##  experiment basic setting
parser.add_argument('--gpu', type=str, default='1,2,3',
                    help='Set CUDA_VISIBLE_DEVICES')
parser.add_argument('--exp_id', type=int, default='1',
                    help='experiment id')

## env revenue record setting
parser.add_argument('--revenue_record_start', type=int, default=100000,#0001,
                    help='record mechanism revenue start epoch (default the same to the exploration epoch)')

parser.add_argument('--revenue_averaged_stamp', type=int, default=100,
                    help='revenue averaged on how many epochs ')

#
fixed_agt_name = 'player_10'

rl_lib_algorithm = ['DQN']


def main(args):
    ## create env
    test_policy = Policy(allocation_mode='highest', payment_mode=args.mechanism)

    env = customed_env(args,policy=test_policy)
    env.reset()
    ##

    print(env.agents)

    test_policy.assign_agent(env.agents)

    agt_list = {agent: None for agent in env.agents}

    # generate agent profile
    for agt in env.agents:

        if agt == fixed_agt_name:
            new_agent = deepcopy(fixed_agent(agent_name=agt, highest_value=args.valuation_range - 1))  # [0- (H-1)]
        else:
            new_agent = deepcopy(greedy_agent(agent_name=agt, highest_value=args.valuation_range - 1))

            new_agent.set_algorithm(
                EpsilonGreedy_auction(bandit_n=args.valuation_range * args.bidding_range,
                                      bidding_range=args.bidding_range, eps=0.01,
                                      start_point=int(args.exploration_epoch),
                                      # random.random()),
                                      overbid=args.overbid, step_floor=10000,
                                      )  # 假设bid 也从分布中取
            )

        agt_list[agt] = new_agent
    ####

    epoch = 0

    # env revenue record init
    revenue_record = env_record(record_start_epoch=args.revenue_record_start,
                                averaged_stamp=args.revenue_averaged_stamp)
    #
    for agent_name in env.agent_iter():

        observation, reward, done, info = env.last()
        # env.render()  # this visualizes a single game
        obs = observation['observation']

        agt = agt_list[agent_name]

        # record
        if obs == args.bidding_range + 1:  # the first round not compute reward
            print('first round')

        else:
            allocation = info['allocation']
            true_value = agt.get_latest_true_value()
            final_reward = compute_reward(true_value, pay=reward, mode='individual',
                                          allocation=allocation)  # reward shaping

            last_action = agt.get_latest_action()
            last_state = []  # [last_true_value, last_action]  # [1-H, 0-H-1]
            agt.update_policy(state=last_state, reward=final_reward)

            revenue_record.record_revenue(allocation=allocation, pay=reward, epoch=int(epoch / args.player_num))

        # new round behavior
        agt.generate_true_value()
        next_true_value = agt.get_latest_true_value()
        new_action = agt.generate_action(obs)
        ###
        env.step(new_action)
        ## print setting
        if(args.print_key):
            print_agent_action(agent_name, fixed_agt_name, new_action, true_value=next_true_value, H=args.bidding_range - 1)
        ##

        epoch += 1

        if epoch ==5:
            test_policy.assign_payment_mode(payment_mode='first_price')

        if ((epoch / args.player_num) % args.estimate_frequent == 0) and (agent_name==env.agents[-1]) :
            estimations = {agent: [] for agent in env.agents}
            estimations = build_estimation(args, agent_list=agt_list, env_agent=env.agents, estimation=estimations)

            plot_figure(path='./results', folder_name=str(args.exp_id), args=args, epoch=int(epoch / args.player_num),
                        prefix=None,
                        estimations=estimations)
            #
            revenue_record.plot_avg_revenue(
                                            args,path='./results',folder_name=str(args.exp_id),
                                             figure_name=get_figure_name(args,epoch=int(epoch / args.player_num)),
                                            mechanism_name=get_mechanism_name(args)
            )
            print(1)
            #
        # env.render()  # this visualizes a single game


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(args)

    main(args)

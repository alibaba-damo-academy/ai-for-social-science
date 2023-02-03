"""A simple multi-agent env with two agents playing rock paper scissors.

This demonstrates running the following policies in competition:
    (1) heuristic policy of repeating the same move
    (2) heuristic policy of beating the last opponent move
    (3) LSTM/feedforward PG policies
    (4) LSTM policy with custom entropy loss
"""

import argparse
import os
from pettingzoo.classic import rps_v2
import random
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
from ray.rllib.algorithms.registry import get_algorithm_class
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples.policy.rock_paper_scissors_dummies import (
    BeatLastHeuristic,
    AlwaysSameHeuristic,
)
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env
from customed_env_auction_deep import *
import matplotlib.pyplot as plt

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=2000, help="Number of iterations to train."
)

parser.add_argument(
    "--algorithm",
    choices=["DQN", "PPO"],
    default="DQN",
    help="The DL algorithm.",
)

parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument('--gpu', type=str, default='1,2,3',
                    help='Set CUDA_VISIBLE_DEVICES')
parser.add_argument(
    "--stop-reward",
    type=float,
    default=1000.0,
    help="Reward at which we stop training.",
)
parser.add_argument('--player_num', type=int, default='4',
                    help='Set agent number')
parser.add_argument('--second_price', type=bool, default=False,
                    help='Set second price or not')
parser.add_argument('--estimate_frequent', type=int, default=100,
                    help='estimate_frequent')

parser.add_argument('--folder_name', type=str, default='deep',
                    help='store_figure path(should be made by yourself XD)')
parser.add_argument('--print_key', type=bool, default=False,
                    help='whether to print procedure')

parser.add_argument('--bidding_range', type=int, default=100,
                    help='whether to print procedure')

#bidding_range = 100

learn_init_list=[100002]
training_env_step = 100

#second_price=True

overbid=True
same_valuation = False

def env_creator(player_num=2,action_space_num=100,second_price=True):
    env = customed_env(player_num=player_num, action_space_num=action_space_num, second_price=second_price,NUM_ITERS =training_env_step)
    env.reset()
    return env





def encoded_input_emb(action,action_space_num=100):
    """
    ???  Meaning of the encoding
    """
    res=torch.zeros(1,action_space_num+1)
    res[0][action]=1
    return res

def test_policy(algo,test_num=10,player_num=2, action_space_num=100, second_price=True):
    test_env = customed_env(player_num=player_num, action_space_num=action_space_num,
                            second_price=second_price,NUM_ITERS =test_num)
    test_env.reset()
    import copy
    agt_list = copy.deepcopy(test_env.agents)
    action_his = {agent: [] for agent in test_env.agents}
    true_value_his = {agent: [] for agent in test_env.agents}
    reward_his = {agent: [] for agent in test_env.agents}

    for agent in test_env.agent_iter():
        observation, reward, done, info = test_env.last()
        # policy_name="policy_"+str(agent[-1])

        action, _, _ = algo.get_policy(agent).compute_single_action(
            encoded_input_emb(observation,action_space_num=action_space_num)
        )
        # print(agent + 'his true value is ' + str(observation) + 'his estimate behavior is ' + str(
        #     action) + 'his done is ' + str(done))
        if done:
            test_env.step(None)
        else:
            test_env.step(action)
            action_his[agent].append(action)
            true_value_his[agent].append(observation)

        # record history

        reward_his[agent].append(reward)

    for agent in agt_list:
        reward_his[agent] = reward_his[agent][1:]  # remove the first 0
        #print('???')
        print(str(agent) + ' avg reward is ' + str(sum(reward_his[agent]) * 1.0 / len(reward_his[agent])))
        print(str(agent) + ' true value and  history is ')

        print(true_value_his[agent])
        print(action_his[agent])
        print('----reward is ')
        print(reward_his[agent])
    #print('!!')


def test_estimation(algo,plot=False,epoch=0,folder_name='deep',
                    player_num=2, action_space_num=100, second_price=True):
    test_env = customed_env(player_num=player_num, action_space_num=action_space_num, second_price=second_price)
    test_env.reset()
    import copy
    agt_list = copy.deepcopy(test_env.agents)
    estimations = {agent: [] for agent in test_env.agents}

    for agent in agt_list:

        # explore all the estimation on observation

        for observation in range(action_space_num):
            action, _, _ = algo.get_policy(agent).compute_single_action(
                encoded_input_emb(observation,action_space_num=action_space_num)
            )

            if observation==0:
                estimations[agent].append(0)
            else:
                estimations[agent].append( action*1.0 / observation )


    if plot:

        saved_data = []
        # bid / value
        if overbid:
            overbid_fig_name = 'overbid'
        else:
            overbid_fig_name = 'no_overbid'

        if same_valuation:
            same_valuation_name = 'Same_value_'
        else:
            same_valuation_name = 'not_Same_value_'



        for agent in agt_list:


            plot_data = estimations[agent]

            saved_data.append(plot_data)
            plt.plot([x for x in range(len(plot_data))], plot_data, label=str(agent))

        plt.ylim(0,2.0)
        #y_ticks=np.arange(0,2,0.2)
        #plt.yticks(y_ticks)
        plt.legend()
        print('./' + folder_name + '/' + same_valuation_name + overbid_fig_name + 'test_players' + str(
            player_num) + '  epoch ' + str(epoch))
        plt.savefig('./' + folder_name + '/' + same_valuation_name + overbid_fig_name + 'test_players' + str(
            player_num) + '  epoch ' + str(epoch))
        plt.show()
        plt.close()


def build_policy_name(agent_num =2,args=None):
    res={}
    for i in range(agent_num):
        name = 'player_'+str(i)
        res[name]=PolicySpec(
                    config={
                        "framework": args.framework,
                    })
    return res

def run_same_policy(args, stop,estimate_frequent=100):
    """Use the same policy for both agents (trivial case)."""


    # obs_space = test_env.observation_space
    # print(obs_space)
    # act_space = test_env.action_spaces

    # config["exploration_config"] = {
    #     # The Exploration class to use.
    #     "type": "EpsilonGreedy",
    #     # Config for the Exploration class' constructor:
    #     "initial_epsilon": 0.1,
    #     "final_epsilon": 0.0,
    #     "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    # }

    config = {
        "env": "second_auction",
        "framework": args.framework,
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "4")),
        "num_workers": 0,
        "num_envs_per_worker": 4,
        "rollout_fragment_length": 10,
        "train_batch_size": 200,
        "metrics_num_episodes_for_smoothing": 200,
        "multiagent":  {
         "policies": build_policy_name(agent_num =args.player_num,args=args),
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


    cls = get_algorithm_class(args.algorithm) if isinstance(args.algorithm, str) else args.algorithm
    algo = cls(config=config)




    for i in range(args.stop_iters):
        print("== Iteration", i, "==")
        results = algo.train()

        # Timesteps reached.

        # test the performances

        #test_env = customed_env(player_num=player_num, action_space_num=bidding_range, second_price=second_price)
        if i % estimate_frequent==0:
            print(pretty_print(results))
            test_policy(algo=algo,player_num=args.player_num,
                        action_space_num=args.bidding_range,
                        second_price=args.second_price)
            test_estimation(algo=algo,plot=True,epoch=i,
                            folder_name=args.folder_name,
                            action_space_num=args.bidding_range,
                            player_num=args.player_num,second_price=args.second_price)
            print('finish estimate')
    #
    # results = tune.Tuner(
    #     "DQN", param_space=config, run_config=air.RunConfig(stop=stop, verbose=1)
    # ).fit()

    if args.as_test:
        # Check vs 0.0 as we are playing a zero-sum game.
        check_learning_achieved(results, 0.0)



if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(args)
    ray.init()

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    register_env("second_auction", lambda config: PettingZooEnv(env_creator(player_num=args.player_num,
                                                                            action_space_num=args.bidding_range,
                                                                            second_price=args.second_price)))
    run_same_policy(args, stop=stop, estimate_frequent=args.estimate_frequent)
    print("run_same_policy: ok.")


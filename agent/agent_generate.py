from copy import deepcopy
from .normal_agent import *
from .signal_agent import *
from env.customed_env_auction import *
from rl_utils.solver import *
from rl_utils.deep_solver import *
import torch
from agent_utils.signal import *

fixed_agt_name = 'player_10'

def generate_agent(env, args, agt_list, budget):
    for i, agt in enumerate(env.agents):
        if args.public_signal:
            highest_value=max_signal_value(args)

        elif (args.assigned_valuation is None) or (i >= len(args.assigned_valuation)):
            highest_value = args.valuation_range - 1

        else:
            highest_value = args.assigned_valuation[i] - 1

        if agt == fixed_agt_name:
            print('build fixed agent on ' + str(agt))
            new_agent = deepcopy(fixed_agent(agent_name=agt, highest_value=highest_value,item_num=args.item_num))  # [0- (H-1)]
        else:
            if args.item_num > 1: 
                print('build multi-item auction agent on ' + str(agt))
                new_agent = deepcopy(greedy_agent(agent_name=agt, highest_value=highest_value, item_num = args.item_num))
            elif args.mechanism == 'tmp':
                print('build dived based greedy agent on ' + str(agt))
                new_agent = deepcopy(dived_greedy_agent(agent_name=agt, highest_value=highest_value))
            elif args.budget_sampled_mode > 0:
                print('build budget greedy agent on ' + str(agt))
                new_agent = deepcopy(greedy_agent_budget_sample_version(agent_name=agt, highest_value=highest_value, \
                                                                        budget_range=budget.get_budget_range(agt)))

            elif args.communication == 1 and (i in args.cm_id):

                new_agent = deepcopy(greedy_agent_cm_version(agent_name=agt, highest_value=highest_value, \
                                                             bidding_range=args.bidding_range, \
                                                             cm_number=len(args.cm_id) - 1, mode='max_only'))
                print('build communication agent on ' + str(agt))

            elif args.public_signal ==1 :

                new_agent = deepcopy(signal_agent(agent_name=agt,
                                                  obs_public_signal_dim=args.agt_obs_public_signal_dim,
                                                  max_public_dim=args.public_signal_dim,
                                                  public_signal_range=args.public_signal_range,
                 private_signal_generator=None))
                #set the
                if args.value_to_signal:
                    print('in the '+str(i)+'agt ' + str(agt) + ' set his obs dim to ' +str(args.agt_obs_public_signal_dim*i) + '-> '+ str(args.agt_obs_public_signal_dim*(i+1)-1))
                    if args.public_signal_dim <= args.agt_obs_public_signal_dim*(i+1)-1:
                        print('error! please enlarge the public signal dim so as to each agent allow isolated dim')
                        print('or change the agent observed dim formulate in agent_generate.py')
                        print(1/0)
                    else:
                        new_agent.set_public_signal_dim(observed_dim_list=[args.agt_obs_public_signal_dim*i+j for j in range(args.agt_obs_public_signal_dim)])

                elif 'kl' in args.folder_name:
                    new_agent.set_public_signal_dim(observed_dim_list=[i])
                else:
                    new_agent.set_public_signal_dim(rnd=True)


                print('build signal based greedy agent on ' + str(agt) +' with observed public signal dim:')
                print(new_agent.get_public_signal_dim_list())


            else:
                print('build normal greedy agent on ' + str(agt))
                new_agent = deepcopy(greedy_agent(agent_name=agt, highest_value=highest_value, item_num = args.item_num))
            
            
            
            if args.algorithm =='deep':
                print('apply the deep learning altgorithm')
                new_agent.set_algorithm(
                    deep_solver( bandit_n=(highest_value + 1) * args.bidding_range,
                                          bidding_range=args.bidding_range, eps=0.1,
                                          start_point=int(args.exploration_epoch),
                                          # random.random()),
                                          overbid=args.overbid, step_floor=int(args.step_floor),
                                          signal=args.public_signal,
                                 cumulative_round=args.item_num-1,state_range=(highest_value + 1),
                                 lr=args.lr, discount_factor=args.multi_item_decay,
                                 model_name='DQN', update_frequent=args.update_frequent,
                                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                          )  # 假设bid 也从分布中取
                    , item_num=args.item_num
                )

            else:
                new_agent.set_algorithm(
                    EpsilonGreedy_auction(bandit_n=(highest_value + 1) * args.bidding_range,
                                          bidding_range=args.bidding_range, eps=0.01,
                                          start_point=int(args.exploration_epoch),
                                          # random.random()),
                                          overbid=args.overbid, step_floor=int(args.step_floor),
                                          signal=args.public_signal
                                          )  # 假设bid 也从分布中取
                    , item_num=args.item_num
                )


            if args.public_bidding_range_asym: 
                EpsilonGreedy_auction(bandit_n=(highest_value + 1) * args.bidding_range,
                                      bidding_range=args.bidding_range/args.player_num*(i+1), eps=0.01,
                                      start_point=int(args.exploration_epoch),
                                      # random.random()),
                                      overbid=args.overbid, step_floor=int(args.step_floor),signal=args.public_signal
                                      )  # 假设bid 也从分布中取
                





        agt_list[agt] = new_agent

    # speical assignment

    if 'kl_sp' in args.folder_name:
        agt_list = special_assign(agt_list,agt_name=args.speicial_agt,args=args)
        print(agt_list[args.speicial_agt].public_signal_dim_list)
    print("agt_list", agt_list)
    return agt_list

def special_assign(agt_list,agt_name='',args=None):

    if agt_name not in agt_list:
        print(agt_name + ' not in agt list')
        print(agt_list)
        print(1/0)
    print('begin speicial assignment')
    #special assign for assymetric user information where agt 0 poseess different amount of information
    new_agent = deepcopy(signal_agent(agent_name=agt_name,
                                      obs_public_signal_dim=2, #observe more information in special
                                      max_public_dim=args.public_signal_dim,
                                      public_signal_range=2*args.public_signal_range,
                                      private_signal_generator=None))

    new_agent.set_public_signal_dim(observed_dim_list=[0,1]) # 2 observation

    highest_value = (2*args.public_signal_range+1)**2


    new_agent.set_algorithm(
        EpsilonGreedy_auction(bandit_n=(highest_value + 1) * args.bidding_range,
                              bidding_range=args.bidding_range, eps=0.01,
                              start_point=int(args.exploration_epoch),
                              # random.random()),
                              overbid=args.overbid, step_floor=int(args.step_floor), signal=args.public_signal
                              )  # 假设bid 也从分布中取
    )
    agt_list[agt_name] = new_agent

    return agt_list
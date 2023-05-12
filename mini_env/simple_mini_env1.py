from env.customed_env_auction import *
from env.multi_item_static_env import *
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
from env.multi_dynamic_env import *
from .mini_env_utils import *

from copy import deepcopy



def bind(instance, func, as_name=None,print_flag=True):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the 
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    if print_flag:
        print('bound_method', bound_method)
    return bound_method


class simple_mini_env1(dynamic_env):
    def __init__(self, round, item_num, mini_env_iters, agent_num, args, policy,
                 set_env=None, init_from_args=True,public_signal_generator=None,
                 public_signal_to_value=None,budget=None,winner_only=True,skip=True
                 ):
        super(simple_mini_env1, self).__init__()

        self.winner_only = winner_only  # only winner knows the true value

        # global setting
        self.round = round  # K
        self.mini_item = item_num  # T
        self.agent_num = agent_num
        self.total_step = round * mini_env_iters  # total step = K*T
        self.args = args

        self.agent_name_list = []  # global agent name list

        self.custom_env = set_env

            # mini env config
        self.mini_env_iters = mini_env_iters  # mini env multple round
        self.mini_player_num = agent_num
        self.mini_action_space = None
        self.mini_policy = policy  # the policy for the allocation and payment rule
        self.mini_agent_name_list = None

        self.mini_env_reward_list=None

            # mini env record


        self.current_step = 0
        self.current_round = 0


        if skip:
            if init_from_args:
                self.init_mini_env_setting_from_args()
                self.init_custom_env()

            #may be none
            if self.check_mini_env():
                self.custom_env.reset()
                # get agent name
                self.mini_agent_name_list = self.custom_env.agents
                self.agent_name_list = deepcopy(self.custom_env.agents)  # global = current on init
                #
                self.mini_policy.assign_agent(
                    self.mini_agent_name_list)  # assign the list of agent into the allocation and payment rule

                self.agt_list = {agent: None for agent in self.custom_env.agents}

                # init reward list
                self.init_reward_list()

            self.init_global_rl_agent()
            self.assign_generate_true_value(self.agent_name_list)

            self.init_record()

    def reset_env(self):
        self.custom_env.reset()
        self.mini_agent_name_list = self.custom_env.agents
        self.mini_policy.assign_agent(
            self.mini_agent_name_list)  # assign the list of agent into the allocation and payment rule


    def init_mini_env_setting_from_args(self):
        if self.mini_item == 1:
            print('setting for mini env with only 1 item with multple round for training')
            self.mini_player_num = self.args.player_num
            self.mini_env_iters = self.args.env_iters
            if self.args.action_mode == 'div':
                self.mini_action_space = self.args.bidding_range * self.args.valuation_range
            else:
                self.mini_action_space = self.args.bidding_range
        else:
            # multi item env: (Assume simulatenous setting)
            print('setting remained for multi-item env')
            self.mini_player_num = self.args.player_num
            self.mini_env_iters = self.args.env_iters
            if self.args.action_mode == 'div':
                self.mini_action_space = self.args.bidding_range * self.args.valuation_range
            else:
                self.mini_action_space = self.args.bidding_range

    def init_custom_env(self):
        """
        The env function often wraps the environment in wrappers by default.
        You can find full documentation for these methods
        elsewhere in the developer documentation.
        """
        if self.mini_item == 1:
            print('init custom env with Single item setting in the round :' + str(self.current_round))

            env = fixed_env_auction(player_num=self.mini_player_num,
                                    action_space_num=self.mini_action_space,
                                    env_iters=self.mini_env_iters,
                                    policy=self.mini_policy
                                    )
            # This wrapper is only for environments which print results to the terminal
            env = wrappers.CaptureStdoutWrapper(env)
            # this wrapper helps error handling for discrete action spaces
            env = wrappers.AssertOutOfBoundsWrapper(env)
            # Provides a wide vareity of helpful user errors
            # Strongly recommended
            env = wrappers.OrderEnforcingWrapper(env)
        else:
            #
            print('init custom env with Multi item setting in the round :' + str(self.current_round))
            env = multi_static_env(player_num=self.mini_player_num,
                            action_space_num=self.mini_action_space,
                            item_num = self.mini_item,
                            env_iters=self.mini_env_iters,
                            policy=self.mini_policy)
            # This wrapper is only for environments which print results to the terminal
            env = wrappers.CaptureStdoutWrapper(env)
            # Provides a wide vareity of helpful user errors
            # Strongly recommended
            env = wrappers.OrderEnforcingWrapper(env)



        self.custom_env = env

        return

    def init_record(self):
        # Record the simulation results
        self.revenue_record = None
        self.revenue_record = env_record(record_start_epoch=self.args.revenue_record_start,  # start record env revenue number
                                    averaged_stamp=self.args.revenue_averaged_stamp)

        # add agent plot
        self.revenue_record.init_agent_record(agent_name_list=self.agent_name_list)

    def init_reward_list(self):
        if self.agent_name_list is not None:
            self.mini_env_reward_list = {agent: 0 for agent in self.agent_name_list} # for global agent name, not the mini env agent name

        else:
            print('the agent name list is None, should init env first ')


    def init_global_rl_agent(self):
        if self.agt_list is not None:
            highest_value = self.args.valuation_range - 1
            for i, agt in enumerate(self.custom_env.agents):
                new_agent = deepcopy(simple_greedy_agent(
                    args=self.args, agent_name=agt, highest_value=highest_value))
                self.agt_list[agt] = new_agent

        else:
            print('agent list is None, init RL agents failed')

    def assign_generate_true_value(self, agent_name_list):

        def generate_true_value(self):
            true_value = random.randint(self.lowest_value, self.highest_value)
            self.record_true_value(true_value)

        for name in agent_name_list:
            bind(self.agt_list[name], generate_true_value,print_flag=False)
            print(f'bind (function) generate_true_value for agent {name}')

    def set_rl_agent(self,rl_agent,agent_name):
        # assign or update an agent(rl_agent) named 'agent_name' to the rl list
        if self.agt_list is not None:
            if agent_name in self.agt_list:
                self.agt_list[agent_name]=rl_agent
                print(f'replace rl agent {agent_name} success!')

        else:
            print('agent list is None, add RL agent' +str(agent_name)+ 'failed')

    def set_rl_agent_algorithm(self, algorithm, agent_name):
        agt = self.agt_list[agent_name]
        agt.set_algorithm(algorithm)
        print(f'set (element) algorithm of agent {agent_name} success!')

    def set_rl_agent_generate_action(self, generate_action, agent_name):
        agt = self.agt_list[agent_name]
        bind(agt, generate_action)
        # agt.set_generate_action(generate_action)
        print(f'binding (function) generate_action with agent {agent_name} success!')

    def set_rl_agent_update_policy(self, update_policy, agent_name):
        agt = self.agt_list[agent_name]
        bind(agt, update_policy)
        # agt.set_update_policy(update_policy)
        print(f'set (function) update_policy of agent {agent_name} success!')

    def set_rl_agent_receive_observation(self, receive_observation, agent_name):
        agt = self.agt_list[agent_name]
        bind(agt, receive_observation)
        # agt.set_receive_observation(receive_observation)
        print(f'set (function) receive_observation of agent {agent_name} success!')

    #add test policy
    def set_rl_agent_test_policy(self,test_policy,agent_name):
        agt = self.agt_list[agent_name]
        bind(agt, test_policy, as_name='test_policy')
        # agt.set_update_policy(update_policy)
        print(f'set (function) test_policy of agent {agent_name} success!')

    def set_mini_env_iter(self, adjusted_mimi_env_iters=None):
        if adjusted_mimi_env_iters is not None:
            self.mini_env_iters = adjusted_mimi_env_iters
            # call the adjust env
            if self.check_mini_env():
                self.init_custom_env()
                self.reset_env()

    def set_mini_env_action_space(self, adjusted_mimi_action_space=None):
        if adjusted_mimi_action_space is not None:
            self.mini_action_space = adjusted_mimi_action_space
            # call the adjust env
            if self.check_mini_env():
                self.init_custom_env()
                self.reset_env()

    def set_mini_env_agent(self, new_agent_name_list=None):
        if new_agent_name_list is not None:
            self.mini_agent_name_list = new_agent_name_list
            if self.mini_policy is not None:
                self.mini_policy.assign_agent(self.mini_agent_name_list)
                print('adjust the involved agent:')
                print(new_agent_name_list)

    def set_mini_env_policy(self,adjusted_policy=None):
        if adjusted_policy is not None:
            self.mini_policy = adjusted_policy
            if self.check_mini_env():
                self.init_custom_env()
                self.reset_env()
                self.mini_policy.assign_agent(self.mini_agent_name_list)  # assign the list of agent into the allocation and payment rule

    def add_reward_by_agt_name(self,agt_name,reward):
        if agt_name in self.mini_env_reward_list:
            self.mini_env_reward_list[agt_name]+=reward

    def self_play(self,agent_name):
        # init env to set agent self play on the env to init rl agent policy

        # the name of agent should be adjusted ->env name vs rl agent name vs self-play agent name
        print('to be done')


    def user_communication(self,env,args,agt_list,agent_name):
        # may move to the basic mini_env class later
        defalut_parm=1
        if defalut_parm:
            extra_info = communication(env, args, agt_list, agent_name)

        return extra_info


    def step(self,agents_list=None):
        # should be adjust to given agent list while others remained silence
        if self.current_round  %self.args.estimate_frequent==0:
            print('round is ' + str(self.current_round))
        # begin K round in single round

        env = self.custom_env
        self.current_step=0
        args=self.args

        supported_function = ['CRRA', 'CARA']
        public_signal=None

        for agent_name in env.agent_iter():
            #print(agent_name)

            observation, reward, termination, truncation, info = env.last()
            
            # env.render()  # this visualizes a single game
            _obs = observation['observation']
            agt = self.agt_list[agent_name]
            agt_idx = env.agents.index(agent_name)

            if (not termination) and (not truncation):
                agt.generate_true_value()
                # print(agt.get_latest_true_value())

            if (self.mini_item == 1 and ((args.action_mode == 'div' and _obs == args.bidding_range * args.valuation_range + 1) or \
                        (args.action_mode == 'same' and _obs == args.bidding_range + 1))) or \
                    (self.mini_item > 1 and ((args.action_mode == 'div' and _obs == [args.bidding_range * args.valuation_range + 1]*self.mini_item) or \
                    (args.action_mode == 'same' and _obs == [args.bidding_range + 1]*self.mini_item))):
                    # the first round not compute reward
                    # In real scenario, signal realization/auction setup happens at the
                    # start of a auction, and auction allocation happens at the end. We
                    # simply the implementation by setting both action to the start by
                    # skipping them at the first round.
                    print('first round skipping')
                    allocation=None #added fixed bug
                    pass

            else:  # Compute reward and update agent policy each round
                allocation = info['allocation']
                # get true value for computing the reward

                # obs record
                obs = agt.get_last_obs()

                if (not termination) and (not truncation):
                    true_value = agt.get_last_round_true_value()
                else: #done, not generate new value
                    true_value = agt.get_latest_true_value()
                
                # adjust each agent with his reward function
                final_reward = compute_reward(true_value=true_value, pay=reward,
                                              allocation=allocation,
                                              reward_shaping=check_support_reward_shaping(supported_function,
                                                                                          args.reward_shaping),
                                              reward_function_config=build_reward_function_config(args),
                                              user_id=agent_name, budget=self.budget,
                                              args=args, info=info  # compute based on the former budget
                                              )  # reward shaping

                last_action = agt.get_latest_action()

                last_state = []  # [last_true_value, last_action]  # [1-H, 0-H-1]
                # update poicy
                agt.update_policy(obs=obs, reward=final_reward,done=truncation)
                agt.record_reward(final_reward)
                # record each step

                self.add_reward_by_agt_name(agt_name=agent_name,reward=final_reward)
                record_step(args, agt, allocation, agent_name, self.current_step+self.mini_env_iters*self.current_round, env, reward, self.revenue_record,agents_list=self.agent_name_list)

            obs = agt.receive_observation(args, self.budget, agent_name, agt_idx, 
                                          extra_info=None, observation=observation,
                                          public_signal=public_signal,
                                          true_value_list=agt.get_true_value_history(), 
                                          action_history=agt.get_action_history(), 
                                          reward_history=agt.get_reward_history(), 
                                          allocation_history=agt.get_allocation_history()
                                          )
            if not self.winner_only or allocation:
                # add the information of the payment into the obs
                obs = increase_info_to_obs(obs, extra_info_name='payment', value=reward)



            agt.record_obs(obs)
            # print('received obs', obs)

            # new round behavior
            # agt.generate_true_value()

            next_true_value = agt.get_latest_true_value()

            if termination or truncation:
                env.step(None)
            else:
                # print(agent_name, 'agent_name')
                new_action = agt.generate_action(obs)  # get the next round action based on the observed budget
                agt.record_action(new_action)
                # print('obs=', obs, 'new_action=', new_action)

                ## cooperate infomation processing [eric]
                submit_info_to_env(args, agt_idx, self.mini_policy, agent_name, next_true_value)
                env.step(new_action)
            ## print setting
            self.current_step += 1

            # require more work here 2.20
            print_process(env, args, self.agt_list, new_action, agent_name, next_true_value, self.current_step+self.mini_env_iters*self.current_round, self.revenue_record,agents_list=self.agent_name_list)

            #env.render()  # this visualizes a single game

        self.current_round+=1

        #print(self.mini_env_reward_list)
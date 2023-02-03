from .customed_env_auction import *
from .multi_item_static_env import *
from agent.normal_agent import *
from rl_utils.solver import *
from rl_utils.reward_shaping import *
from rl_utils.budget import *
from rl_utils.communication import *
from rl_utils.util import *
from plot.plot_util import *
from plot.print_process import *
from .env_record import *
from .policy import *
from agent_utils.action_encoding import *
from agent_utils.action_generation import *
from agent_utils.agent_updating import *
from agent_utils.obs_utils import *
from agent_utils.signal import *
from agent_utils.valuation_generator import *
from .multi_dynamic_env import *
from agent.agent_generate import *

from copy import deepcopy


class self_play_env(dynamic_env):
    def __init__(self, item_num, mini_env_iters, imaged_agent_num,ori_agt_id,
                 args, policy,
                 self_play_agent,estimate_frequent=None,
                 set_env=None, init_from_args=True,sequential_train=False, public_signal_generator=None,
                 public_signal_to_value=None, budget=None
                 ):
        super(self_play_env, self).__init__()
        # self play setting

        self.mini_item = item_num  # T
        self.imaged_agent_num = imaged_agent_num
        self.total_step = mini_env_iters  # total step = K*T
        self.args = args


        self.sequential_train=sequential_train
        self.ori_agt_id=ori_agt_id
        self.self_play_agent=self_play_agent


        self.custom_env = set_env
        # mini env setting
        # mini env value config
        self.public_signal_generator = public_signal_generator
        self.public_signal_to_value = public_signal_to_value
        self.budget = budget

        # mini env config
        self.mini_env_iters = mini_env_iters  # mini env multple round
        self.new_estimate_frequent=estimate_frequent


        self.mini_action_space = None
        self.mini_policy = policy  # the policy for the allocation and payment rule
        self.mini_agent_name_list = None

        self.mini_env_reward_list = None

        # mini env record

        self.current_step = 0
        self.current_round = 0

        if init_from_args:
            self.recheck_args_in_self_play() #some settings in args can be varied such as infos and budgets

            self.init_mini_env_setting_from_args()
            self.init_custom_env()

        if self.check_mini_env():
            self.custom_env.reset()
            # get agent name
            self.mini_agent_name_list = self.custom_env.agents
            self.agent_name_list = deepcopy(self.custom_env.agents)  # global = current on init
            #
            self.mini_policy.assign_agent(
                self.mini_agent_name_list)  # assign the list of agent into the allocation and payment rule

            self.init_generator_from_args()
            self.agt_list = {agent: None for agent in self.custom_env.agents}

            # init reward list
            self.init_reward_list()


        #
        #self.init_global_rl_agent()
        self.init_self_play_rl_agent()



        self.init_record()

    def recheck_args_in_self_play(self):
        ##
        #  some settings in original args setting should be re-orgnaized in self play
        #  such as budget setting and the infos setting
        #  we set the budget first and remained infos later
        ##

        #update print frequent
        if self.new_estimate_frequent is not None:
            self.args.estimate_frequent=self.new_estimate_frequent
        self.args.player_num=self.imaged_agent_num

        # budget setting re-org
        if self.args.budget_param is None :
            return
        else:
             if self.ori_agt_id < len(self.args.budget_param):
                ori_budget_param = self.args.budget_param[self.ori_agt_id]
             else:
                ori_budget_param = self.args.budget_param[-1] #last param as the budget

             if len(self.args.budget_param) > self.imaged_agent_num:  # as args params should be rechecked
                self.args.budget_param=self.args.budget_param[:self.imaged_agent_num]

             #set all the budget to the ori budget
             for i in range(len(self.args.budget_param)):
                self.args.budget_param[i]=ori_budget_param
        return



    def get_trained_agt(self):
        mini_agt_name=self.agent_name_list[0]

        return self.agt_list[mini_agt_name]

    def reset_env(self):
        self.custom_env.reset()
        self.mini_agent_name_list = self.custom_env.agents
        self.mini_policy.assign_agent(
            self.mini_agent_name_list)  # assign the list of agent into the allocation and payment rule


    def init_mini_env_setting_from_args(self):
        if self.mini_item == 1:
            print('setting for mini env with only 1 item with multple round for training')

            #self.mini_env_iters = self.args.env_iters
            if self.args.action_mode == 'div':
                self.mini_action_space = self.args.bidding_range * self.args.valuation_range
            else:
                self.mini_action_space = self.args.bidding_range
        else:
            # multi item env: (Assume simulatenous setting)
            print('setting remained for multi-item env')


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

            env = fixed_env_auction(player_num=self.imaged_agent_num,
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
            env = multi_static_env(player_num=self.imaged_agent_num,
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

    def init_generator_from_args(self):
        args=self.args


        if args.budget_mode is not None:
            budget = build_budget(args, agent_name_list=self.custom_env.agents)

            self.set_budget(budget)

        if args.public_signal:
            if 'kl' in args.folder_name:
                public_signal_generator = KL_Signal(dtype='Discrete', feature_dim=args.public_signal_dim,
                                                    lower_bound=0, upper_bound=args.public_signal_range,
                                                    generation_method='uniform',
                                                    value_to_signal=args.value_to_signal,
                                                    value_generator_mode=args.value_generator_mode
                                                    )  # assume upperbound
            else:
                public_signal_generator = Signal(dtype='Discrete', feature_dim=args.public_signal_dim,
                                                 lower_bound=0, upper_bound=args.public_signal_range,
                                                 generation_method='uniform',
                                                 value_to_signal=args.value_to_signal,
                                                 public_signal_asym=args.public_signal_asym
                                                 )  # assume upperbound
            if args.value_to_signal:
                value_generator_mode = 'fixed'
            else:
                value_generator_mode = args.value_generator_mode

            public_signal_to_value = public_value_generator(signal_generator=public_signal_generator,
                                                            value_generator_mode=value_generator_mode,
                                                            valuation_range=args.valuation_range
                                                            )
            if args.public_signal_spectrum:
                # Set up an interpolation experiment between pure private value and
                # pure public value model
                new_public_signal_generator = deepcopy(public_signal_generator)
                new_public_signal_generator.lower_bound = args.public_signal_lower_bound
                new_public_signal_generator.upper_bound = args.public_signal_range - args.public_signal_lower_bound
                public_signal_to_value = public_value_generator(signal_generator=new_public_signal_generator,
                                                                value_generator_mode=value_generator_mode,
                                                                valuation_range=args.valuation_range
                                                                )
            if public_signal_generator is not None:
                self.set_public_signal_generator(public_signal_generator)
            if public_signal_to_value is not None:
                self.set_public_signal_to_value(public_signal_to_value)

    def init_record(self):
        # Record the simulation results
        self.revenue_record = None
        self.revenue_record = env_record(record_start_epoch=self.args.revenue_record_start,  # start record env revenue number
                                    averaged_stamp=self.args.revenue_averaged_stamp)

    def init_reward_list(self):
        if self.agent_name_list is not None:
            self.mini_env_reward_list = {agent: 0 for agent in self.agent_name_list} # for global agent name, not the mini env agent name

        else:
            print('the agent name list is None, should init env first ')


    def init_global_rl_agent(self):
        if self.agt_list is not None:
            agt_list = generate_agent(env=self.custom_env, args=self.args, agt_list=self.agt_list, budget=self.budget)
            self.agt_list=agt_list

        else:
            print('agent list is None, init RL agents failed')
    def init_self_play_rl_agent(self):
        first=0
        for agt in self.agt_list:
            if first==0:
                deep_copy=False
            else:
                deep_copy=self.sequential_train
            self.set_rl_agent(rl_agent=self.self_play_agent,agent_name=agt,deep_copy=deep_copy)
            print('begin add self_play agt ' + str(first) + ' | named ' + str(agt))

            first+=1


    def set_rl_agent(self,rl_agent,agent_name,deep_copy=False):
        # assign or update an agent(rl_agent) named 'agent_name' to the rl list
        if self.agt_list is not None:
            if agent_name in self.agt_list:
                if deep_copy:
                    #deep copy
                    self.agt_list[agent_name] = deepcopy(rl_agent)

                else:
                    self.agt_list[agent_name]=rl_agent
                print('replace rl agent sucess ')

        else:
            print('agent list is None, add RL agent' +str(agent_name)+ 'failed')


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

    def step(self, agents_list=None):
        # should be adjust to given agent list while others remained silence

        # begin K round in single round

        env = self.custom_env
        self.current_step = 0
        args = self.args

        supported_function = ['CRRA', 'CARA']
        public_signal = None

        for agent_name in env.agent_iter():
            # print(agent_name)

            observation, reward, termination, truncation, info = env.last()

            # env.render()  # this visualizes a single game
            _obs = observation['observation']
            agt = self.agt_list[agent_name]
            agt_idx = env.agents.index(agent_name)

            # generate action for all agent
            if agent_name == env.agents[0] and (not termination) and (not truncation):
                # start of a bidding episode since we loop according to the order of
                # agent

                # generate public signal
                if self.public_signal_generator is not None:
                    last_public_value = self.public_signal_to_value.get_last_public_gnd_value()
                    # public_signal_to_value.generate_value(obs=None, gnd=True, weighted=False) # get value before update the new signal

                    self.public_signal_generator.generate_signal(data_type='int')  # update the signal
                    # for debug
                    global_signal = self.public_signal_generator.get_whole_signal_realization()

                # Update auction info (signal realization) at the start of an auction
                update_agent_profile(env, args, self.agt_list, agent_name,
                                     self.budget,
                                     public_signal_generator=self.public_signal_generator,
                                     public_signal_to_value=self.public_signal_to_value
                                     )  # should update all the latest true value function
                # for debug
                if self.public_signal_generator is not None:
                    global_signal = self.public_signal_generator.get_whole_signal_realization()

                extra_info = communication(env, args, self.agt_list, agent_name)
            if (self.mini_item == 1 and (
                    (args.action_mode == 'div' and _obs == args.bidding_range * args.valuation_range + 1) or \
                    (args.action_mode == 'same' and _obs == args.bidding_range + 1))) or \
                    (self.mini_item > 1 and ((args.action_mode == 'div' and _obs == [
                        args.bidding_range * args.valuation_range + 1] * self.mini_item) or \
                                             (args.action_mode == 'same' and _obs == [
                                                 args.bidding_range + 1] * self.mini_item))):
                # the first round not compute reward
                # In real scenario, signal realization/auction setup happens at the
                # start of a auction, and auction allocation happens at the end. We
                # simply the implementation by setting both action to the start by
                # skipping them at the first round.

                a = 1
            else:  # Compute reward and update agent policy each round
                allocation = info['allocation']
                # get true value for computing the reward
                if args.public_signal:
                    true_value = last_public_value
                    if true_value is None:  # if last round is none = not exist last round
                        true_value = self.public_signal_to_value.get_last_public_gnd_value()

                else:
                    if (not termination) and (not truncation):
                        true_value = agt.get_last_round_true_value()
                    else:  # done, not generate new value
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
                agt.update_policy(state=last_state, reward=final_reward, done=truncation)
                # record each step

                self.add_reward_by_agt_name(agt_name=agent_name, reward=final_reward)

                record_step(args, agt, allocation, agent_name,
                            self.current_step , env, reward,
                            self.revenue_record, agents_list=self.agent_name_list)

            if args.public_signal:
                # Given the public signal realization, generate the public signal
                # observable to the agent
                public_signal = self.public_signal_generator.get_partial_signal_realization(
                    observed_dim_list=agt.get_public_signal_dim_list())
                agt.receive_partial_obs_true_value(
                    true_value=self.public_signal_to_value.generate_value(obs=public_signal))

            obs = receive_observation(args, self.budget, agent_name, agt_idx, extra_info, observation,
                                      public_signal=public_signal
                                      )

            # new round behavior
            # agt.generate_true_value()

            next_true_value = agt.get_latest_true_value()

            if termination or truncation:
                env.step(None)
            else:
                new_action = action_generation(args, agt, obs)  # get the next round action based on the observed budget

                ## cooperate infomation processing [eric]
                submit_info_to_env(args, agt_idx, self.mini_policy, agent_name, next_true_value)
                env.step(new_action)
            ## print setting
            self.current_step += 1
            print_process(env, args, self.agt_list, new_action, agent_name, next_true_value,
                          self.current_step , self.revenue_record,
                          agents_list=self.agent_name_list,prefix='self_play_agt'+str(self.ori_agt_id))

            # env.render()  # this visualizes a single game



        # print(self.mini_env_reward_list)
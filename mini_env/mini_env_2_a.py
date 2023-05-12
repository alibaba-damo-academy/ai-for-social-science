from env.customed_env_auction import *
from env.multi_item_static_env import *
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

from pricing_signal.plat_deliver_signal import *
from agent.signal_pricing_agent import *

from agent_utils.valuation_generator import *

from agent.agent_generate import *
from env.multi_dynamic_env import *
from .mini_env_utils import *

from copy import deepcopy

def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    print('bound_method', bound_method)
    return bound_method



class mini_env_ver_platform_signal(dynamic_env):
    def __init__(self, round, item_num, mini_env_iters, agent_num, args, policy, set_env=None, init_from_args=True,
                 public_signal_generator=None, budget=None,
                 private_signal_generator=None,signal_type='nomral',winner_only=True
                 ):

        super(mini_env_ver_platform_signal).__init__()

        # signal game setting
        self.winner_only = winner_only #only winner knows the true value


        # global setting
        self.round = round  # K
        self.mini_item = item_num  # T
        self.agent_num = agent_num
        self.total_step = round * mini_env_iters  # total step = K*T
        self.args = args

        self.agent_name_list = []  # global agent name list

        self.custom_env = set_env
        # mini env setting
        # mini env value config
        self.public_signal_generator = public_signal_generator

        self.budget = budget

        self.private_signal_generator = private_signal_generator

        self.sealed_auction = True  #
        # self.winner_only = True #only winner know the true value

        self.last_public_signal=None



        # mini env config
        self.mini_env_iters = mini_env_iters  # mini env multple round
        self.mini_player_num = agent_num
        self.mini_action_space = None
        self.mini_policy = policy  # the policy for the allocation and payment rule
        self.mini_agent_name_list = None

        self.mini_env_reward_list = None

        # mini env record

        self.current_step = 0
        self.current_round = 0
        self.saved_signal=None

        if init_from_args:
            self.init_mini_env_setting_from_args()

        if self.custom_env is None:
            self.init_custom_env()

        # may be none

        if self.check_mini_env():
            self.custom_env.reset()
            # get agent name
            self.mini_agent_name_list = self.custom_env.agents
            self.agent_name_list = deepcopy(self.custom_env.agents)  # global = current on init
            #
            self.mini_policy.assign_agent(
                self.mini_agent_name_list)  # assign the list of agent into the allocation and payment rule

            self.generate_default_signal(signal_type=signal_type) # assign the public signal generator

            self.agt_list = {agent: None for agent in self.custom_env.agents}

            # init reward list
            self.init_reward_list()

        self.init_global_rl_agent(default=True,signal_type=signal_type)

        self.init_record()

        print('test')
    def reset_env(self):
        self.custom_env.reset()
        self.mini_agent_name_list = self.custom_env.agents
        self.mini_policy.assign_agent(
            self.mini_agent_name_list)  # assign the list of agent into the allocation and payment rule


    def init_mini_env_setting_from_args(self):
        # load setting from args
            print('setting for mini env with only 1 item with multple round for training')
            self.mini_player_num = self.args.player_num
            self.mini_env_iters = self.args.env_iters

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
                                   item_num=self.mini_item,
                                   env_iters=self.mini_env_iters,
                                   policy=self.mini_policy)
            # This wrapper is only for environments which print results to the terminal
            env = wrappers.CaptureStdoutWrapper(env)
            # Provides a wide vareity of helpful user errors
            # Strongly recommended
            env = wrappers.OrderEnforcingWrapper(env)

        self.custom_env = env

        return
    ## init env

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

    # set func
    def set_env_public_signal_to_value(self, platform_public_signal_to_value):

        bind(self, platform_public_signal_to_value,as_name='platform_public_signal_to_value')

        #self.platform_public_signal_to_value(signal=0)
        # agt.set_generate_action(generate_action)
        print(f'binding (function) platform_public_signal_to_value in env function success!')

    def set_env_private_signal_to_value(self, shared_private_signal_to_value):

        bind(self, shared_private_signal_to_value,as_name='shared_private_signal_to_value')
        # agt.set_generate_action(generate_action)
        print(f'binding (function) private_signal_to_value in env function success!')

    def set_agt_private_signal_to_value(self, private_signal_to_value,agent_name):
        agt = self.agt_list[agent_name]
        bind(agt, private_signal_to_value,as_name='private_signal_to_value')

        # agt.set_generate_action(generate_action)
        print(f'binding (function) private_signal_to_value of agent {agent_name} success!')


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
        bind(agt, generate_action,as_name='generate_action')
        # agt.set_generate_action(generate_action)
        print(f'binding (function) generate_action with agent {agent_name} success!')

    def set_rl_agent_update_policy(self, update_policy, agent_name):
        agt = self.agt_list[agent_name]
        bind(agt, update_policy,as_name='update_policy')
        # agt.set_update_policy(update_policy)
        print(f'set (function) update_policy of agent {agent_name} success!')

    def set_rl_agent_test_policy(self,test_policy,agent_name):
        agt = self.agt_list[agent_name]
        bind(agt, test_policy, as_name='test_policy')
        # agt.set_update_policy(update_policy)
        print(f'set (function) test_policy of agent {agent_name} success!')


    def set_rl_agent_receive_observation(self, receive_observation, agent_name):
        agt = self.agt_list[agent_name]
        bind(agt, receive_observation,as_name='receive_observation')
        # agt.set_receive_observation(receive_observation)
        print(f'set (function) receive_observation of agent {agent_name} success!')


    def set_update_env_public_signal(self,update_env_public_signal):

        bind(self, update_env_public_signal,as_name='update_env_public_signal')

        #self.update_env_public_signal()
        # agt.set_generate_action(generate_action)
        print(f'binding (function) update_env_public_signal in env function success!')

    #end set func


    # env signal -> value function
    def platform_public_signal_to_value(self,signal, func='sum', weights=None):
        #
        # denotes the function for signal->value [v=f(s)]
        #
        if signal is None or len(signal) == 0:
            return 0

        if func == 'sum':
            return sum(signal)
        if func == 'avg':
            return sum(signal) / len(signal)

        if func == 'weighted':
            return weighted_sum(signal, weights)

    def shared_private_signal_to_value(self,signal, func='sum', weights=None):
        #
        # denotes the function for signal->value [v=f(s)]
        #
        if signal is None or len(signal) == 0:
            return 0

        if func == 'sum':
            return sum(signal)
        if func == 'avg':
            return sum(signal) / len(signal)

        if func == 'weighted':
            return weighted_sum(signal, weights)
    ##

    def init_global_rl_agent(self,default=True,signal_type='normal'):
        print('-------------')
        print('init global agent in mini signal game !!')
        print('----------')

        if self.agt_list is not None:
            if default:
                self.generate_default_agent(signal_type=signal_type)
            else:
                print('generate using non-default agt method (to_be_write) ')
            #generate_agent(env=self.custom_env, args=self.args, agt_list=self.agt_list, budget=self.budget)

            # self.agt_list=agt_list

        else:
            print('agent list is None, init RL agents failed')

    def init_agent_obs_dim_list(self,args,agt_id,agt,new_agent,signal_type):
        # set the observed dim
        if args.value_to_signal:
            print('in the ' + str(agt_id) + 'agt ' + str(agt) + ' set his obs dim to ' + str(
                args.agt_obs_public_signal_dim * agt_id) + '-> ' + str(args.agt_obs_public_signal_dim * (i + 1) - 1))

            if args.public_signal_dim <= args.agt_obs_public_signal_dim * (agt_id + 1) - 1:
                print('error! please enlarge the public signal dim so as to each agent allow isolated dim')
                print('or change the agent observed dim formulate in agent_generate.py')
                print(1 / 0)
            else:
                new_agent.set_public_signal_dim(observed_dim_list=[args.agt_obs_public_signal_dim * agt_id + j for j in
                                                                   range(args.agt_obs_public_signal_dim)])
        elif signal_type == 'kl':
            new_agent.set_public_signal_dim(observed_dim_list=[agt_id])
        else:
            # denote as random
            dim_mask = [indexs for indexs in range(args.public_signal_dim)]  # max public dim
            random.shuffle(dim_mask)
            observed_dim_list = dim_mask[:new_agent.obs_public_signal_dim]
            new_agent.set_public_signal_dim(observed_dim_list=observed_dim_list)  # random assign
            ## eric noted!


    def get_args_public_signal_dim(self,args,agent_name):


        if isinstance(args.agt_obs_public_signal_dim,list):
            # represent is list to represent the signal dim where different agent has different signal :
            return args.agt_obs_public_signal_dim[agent_name]
        else:
            return args.agt_obs_public_signal_dim

    def generate_default_agent(self,signal_type='normal'):

        # default use greedy 1 dim signal agent
        env=self.custom_env
        args=self.args

        for i, agt in enumerate(env.agents):
            if args.public_signal:
                highest_value = max_signal_value(args)

            elif (args.assigned_valuation is None) or (i >= len(args.assigned_valuation)):
                highest_value = args.valuation_range - 1

            else:
                highest_value = args.assigned_valuation[i] - 1

            ## default signal agent should be re-rewrite

            new_agent = deepcopy(simple_signal_greedy_agent(args=args,agent_name=agt,
                                              obs_public_signal_dim=self.get_args_public_signal_dim(args,agent_name=agt),
                                              public_signal_range=args.public_signal_range,
                                              private_signal_generator=None))

            # set the observed dim
            self.init_agent_obs_dim_list(args, agt_id=i, agt=agt, new_agent=new_agent, signal_type=signal_type)

            #set algorithm
            new_agent.set_algorithm(
                EpsilonGreedy_auction(bandit_n=(highest_value + 1) * args.bidding_range,
                                      bidding_range=args.bidding_range, eps=0.01,
                                      start_point=int(args.exploration_epoch),
                                      # random.random()),
                                      overbid=args.overbid, step_floor=int(args.step_floor),
                                      signal=args.public_signal
                                      )  # 假设bid 也从分布中取
                #, item_num=args.item_num # item number setting delete
            )
            self.agt_list[agt] = new_agent

    def generate_default_signal(self,signal_type='normal'):
        #signal_type =['normal' | 'kl' ]
        args=self.args

        if signal_type == 'kl' :
            public_signal_generator = KL_Signal(dtype='Discrete', feature_dim=args.public_signal_dim,
                                                lower_bound=0, upper_bound=args.public_signal_range,
                                                generation_method='uniform',
                                                value_to_signal=args.value_to_signal,
                                                value_generator_mode=args.value_generator_mode
                                                )  # assume upperbound

            self.signal_type='kl'

        else:
            public_signal_generator = platform_Signal_v1(dtype='Discrete', feature_dim=args.public_signal_dim,
                                             lower_bound=0, upper_bound=args.public_signal_range,
                                             generation_method='uniform',
                                             value_to_signal=args.value_to_signal,
                                             public_signal_asym=args.public_signal_asym
                                             )  # assume upperbound


        # generate true value from signal function

        if public_signal_generator is not None:
            self.set_public_signal_generator(public_signal_generator)




    def private_signal_to_value(self,agent_name,agt):
        private_value =0
        if self.private_signal_generator is not None:

            private_signal = self.private_signal_generator(agent_name=agent_name)
            if self.args.agt_independent_private_value_func:
                private_value = agt.private_signal_to_value(private_signal)
            else:
                private_value = self.shared_private_signal_to_value(private_signal)
        else:
            private_value = 0 #agt.generate_true_value

        return private_value


    def get_last_agt_true_value_from_signal(self,agt,agent_name,
                                       termination,truncation,
                                       allocation,winner_only=True):


        true_value =0

        if (not termination) and (not truncation):

            private_value = self.private_signal_to_value(agent_name, agt)

            if self.public_signal_generator is not None:

                # the true signal and the true value after purchase
                true_public_signal = self.last_public_signal

                public_value = self.platform_public_signal_to_value(true_public_signal,

                                                                    )

            else:
                public_value=0

            # winner only determine
            if not winner_only or allocation:
                true_value =private_value+public_value
            else:
                #winner only and not allocation
                true_value=private_value


        return true_value

    def update_env_public_signal(self,data_type='int'):
        if self.public_signal_generator is None:
            #not exist signal generator
            print('not exist signal generator, skip signal updating ')
            return


        self.last_public_signal = deepcopy(self.public_signal_generator.get_whole_signal_realization())
        #receive the public signal results and saved

        self.public_signal_generator.generate_signal(data_type=data_type)  # update the signal

        #Full information generated from the platform

        return

    def update_private_signal(self):
        #
        #require to added

        return



    def step(self, agents_list=None):
        # should be adjust to given agent list while others remained silence
        if self.current_round % self.args.estimate_frequent == 0:
            print('round is ' + str(self.current_round))
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

            if agent_name == env.agents[0] :
                #update public signal before first agent move
                # platform movement
                self.update_env_public_signal()

                # require private signal
                self.update_private_signal()


            # get the true value from signal


            if (self.mini_item == 1 and (
                    (args.action_mode == 'same' and _obs == args.bidding_range + 1))) or (self.mini_item > 1 and ((args.action_mode == 'div' and _obs == [
                        args.bidding_range * args.valuation_range + 1] * self.mini_item) or
                                             (args.action_mode == 'same' and _obs == [args.bidding_range + 1] * self.mini_item))):
                # the first round not compute reward
                # In real scenario, signal realization/auction setup happens at the
                # start of a auction, and auction allocation happens at the end. We
                # simply the implementation by setting both action to the start by
                # skipping them at the first round.
                print('first round skipping')
                true_value = 0

                pass

            else:  # Compute reward and update agent policy each round
                allocation = info['allocation']
                # get true value for computing the reward

                true_value = self.get_last_agt_true_value_from_signal(agt, agent_name,

                                                                 termination, truncation, allocation,
                                                                 winner_only=self.winner_only)





                # adjust each agent with his reward function
                final_reward = compute_reward(true_value=true_value, pay=reward,
                                              allocation=allocation,
                                              reward_shaping=check_support_reward_shaping(supported_function,
                                                                                          args.reward_shaping),
                                              reward_function_config=build_reward_function_config(args),
                                              user_id=agent_name, budget=self.budget,
                                              args=args, info=info  # compute based on the former budget
                                              )  # reward shaping



                # obs record?

                obs = agt.get_last_obs()
                obs = increase_info_to_obs(obs, extra_info_name='allocation', value=allocation)

                # add more obs if not winner only
                if not self.winner_only or allocation:
                    obs = increase_info_to_obs(obs,extra_info_name='true_value',value=true_value)
                    obs = increase_info_to_obs(obs,extra_info_name='payment',value=reward)



                # update poicy
                agt.update_policy(obs=obs, reward=final_reward, done=truncation)

                #agt.record_reward(final_reward)
                # record each step

                #self.add_reward_by_agt_name(agt_name=agent_name, reward=final_reward)

                record_step(args, agt, allocation, agent_name,
                            self.current_step + self.mini_env_iters * self.current_round, env, reward,
                            self.revenue_record, agents_list=self.agent_name_list)

            # rebuild the observation

            public_signal = set_agent_signal_observation(
                public_signal_generator = self.public_signal_generator,
                agent_name=agent_name,
                agt=agt,
                mode='default'

            )


            obs = agt.receive_observation(args, self.budget, agent_name, agt_idx,
                                          extra_info=None, observation=observation,
                                          public_signal=public_signal,
                                          # true_value_list=agt.get_true_value_history(),
                                          # action_history=agt.get_action_history(),
                                          # reward_history=agt.get_reward_history(),
                                          # allocation_history=agt.get_allocation_history()
                                          )
            # print('received obs', obs)

            # new round behavior


            # in the signal game, true value is the sum of the latest signal public value + private value
            next_true_value = self.platform_public_signal_to_value(public_signal) + self.private_signal_to_value(agent_name,agt)
            #agt.get_latest_true_value()



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

            self.print_results(env, args, new_action, agent_name, next_true_value)

            # env.render()  # this visualizes a single game

        self.current_round += 1

        # print(self.mini_env_reward_list)
    def print_results(self, env, args, new_action, agent_name, next_true_value):

            use_old = False
            if use_old:
                print_process(env, args, self.agt_list, new_action, agent_name, next_true_value,
                              self.current_step + self.mini_env_iters * self.current_round, self.revenue_record,
                              agents_list=self.agent_name_list)
            else:
                agents_list = self.agent_name_list
                agt_list = self.agt_list
                epoch = self.current_step + self.mini_env_iters * self.current_round
                prefix = None

                if ((epoch / args.player_num) % args.estimate_frequent == 0) and (agent_name == agents_list[-1]):
                    estimations = {agent: [] for agent in agents_list}
                    estimations = self.build_estimation(args, agent_list=agt_list, env_agent=agents_list,
                                                        estimation=estimations)

                    plot_figure(path='./results', folder_name=str(args.exp_id), args=args,
                                epoch=int(epoch / args.player_num),
                                prefix=prefix,
                                estimations=estimations)

                    self.revenue_record.plot_avg_revenue(
                        args, path='./results', folder_name=str(args.exp_id),
                        figure_name=get_figure_name(args, epoch=int(epoch / args.player_num)),
                        mechanism_name=get_mechanism_name(args),
                        # plot_y_range=[0.4*args.valuation_range,args.valuation_range] #plot range
                    )

                    if args.record_efficiency == 1:
                        self.revenue_record.plot_avg_efficiency(
                            args, path='./results', folder_name=str(args.exp_id),
                            figure_name=get_figure_name(args, epoch=int(epoch / args.player_num)),
                            mechanism_name=get_mechanism_name(args),
                            # plot_y_range=[0.4*args.valuation_range,args.valuation_range] #plot range
                        )

                    # print revenue
                    print_agent_reward(agent_list=agt_list, agent_name_list=agents_list,
                                       avg_epoch=args.revenue_averaged_stamp,
                                       filter_allocation=True
                                       )

                    print('the latest avg revenue is ' + str(self.revenue_record.get_last_avg_revenue()))

    def build_estimation(self, args, agent_list, env_agent, estimation):

        #require more modify towards here

            signals = {agent: [] for agent in env_agent}

            for i, agent in enumerate(env_agent):
                # max_public_signal=[]*args.agt_obs_public_signal_dim
                # max_state =agent_list[agent].encode_signal(public_signal=,
                #                                            private_signal=None)
                if 'kl_sp' in args.folder_name and args.speicial_agt is not None and args.speicial_agt == agent:
                    estimation = build_speicial_estimation(args, agent_list, agt_name=agent, estimation=estimation)
                else:
                    max_state = max_signal_value(args)
                    for state in range(max_state):
                        action = agent_list[agent].test_policy(state=state)

                        # define the bid / signal
                        if args.value_to_signal:
                            partial_signal_value = 1.0
                        elif 'kl' in args.folder_name:
                            partial_signal_value = max(1.0, state * 1.0)
                        else:
                            partial_signal_value = signal_to_value_sum(signal_decode(args, state))

                        if partial_signal_value == 0:
                            if args.test_mini_env == 1:
                                estimation[agent].append(action)
                            else:
                                estimation[agent].append(0.0)

                        else:
                            estimation[agent].append(action * 1.0 / partial_signal_value)

            return estimation

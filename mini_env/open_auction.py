from env.customed_open_auction import *
from env.multi_item_static_env import *
from agent.normal_agent import *
from agent.ascending_agent import *
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
from pricing_signal.private_signal_generator import *

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



class open_ascending_auction(dynamic_env):
    def __init__(self, round, item_num, mini_env_iters, agent_num, args, policy, set_env=None, init_from_args=True,
                 public_signal_generator=None, budget=None,
                 private_signal_generator=None,signal_type='nomral',winner_only=True,init_private_signal=False,
                 max_bidding_times=100, minimal_raise = 1

                 ):

        super(open_ascending_auction).__init__()

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
        # mini env  open auction setting


        self.max_bidding_times  = max_bidding_times
        self.minimal_raise = minimal_raise
        self.last_bid_list = None

        # mini env value config
        self.public_signal_generator = public_signal_generator

        self.budget = budget

        self.private_signal_generator = private_signal_generator
        self.init_private_signal = init_private_signal

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
            self.last_bid_list = {agent: 0 for agent in self.agent_name_list} #init record open auction

        self.init_global_rl_agent(default=True,signal_type=signal_type)

        self.init_record()

        print('open auction init ! ')

    def re_init(self,init_from_args=True):
        # restart the whole env including agent
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

            self.generate_default_signal(signal_type=signal_type)  # assign the public signal generator

            self.agt_list = {agent: None for agent in self.custom_env.agents}

            # init reward list
            self.init_reward_list()
            self.last_bid_list = {agent: 0 for agent in self.agent_name_list}  # init record open auction

        self.init_global_rl_agent(default=True, signal_type=signal_type)

        self.init_record()

        print('open auction init ! ')


    def reset_env(self):
        self.custom_env.reset()
        self.mini_agent_name_list = self.custom_env.agents
        self.mini_policy.assign_agent(
            self.mini_agent_name_list)  # assign the list of agent into the allocation and payment rule


        self.last_bid_list = {agent: 0 for agent in self.agent_name_list}

    def init_mini_env_setting_from_args(self):
        # load setting from args
            print('setting for mini env with only 1 item with multple round for training')
            self.mini_player_num = self.args.player_num
            self.mini_env_iters = self.args.env_iters

            self.mini_action_space = self.args.bidding_range

            self.init_private_signal = self.args.private_signal


    def init_custom_env(self):
        """
        The env function often wraps the environment in wrappers by default.
        You can find full documentation for these methods
        elsewhere in the developer documentation.
        """
        if self.mini_item == 1:
            print('init custom opening env with Single item setting in the round :' + str(self.current_round))

            env = open_auction(player_num=self.mini_player_num,
                                    action_space_num=self.mini_action_space,
                                    max_bidding_times=self.max_bidding_times,#mini_env_iters,
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
            print('no found mini env for multi-item setting in open auction')
            # print('init custom env with Multi item setting in the round :' + str(self.current_round))

            # env = multi_static_env(player_num=self.mini_player_num,
            #                        action_space_num=self.mini_action_space,
            #                        item_num=self.mini_item,
            #                        env_iters=self.mini_env_iters,
            #                        policy=self.mini_policy)
            # # This wrapper is only for environments which print results to the terminal
            # env = wrappers.CaptureStdoutWrapper(env)
            # # Provides a wide vareity of helpful user errors
            # # Strongly recommended
            # env = wrappers.OrderEnforcingWrapper(env)

        self.custom_env = env

        return
    ## init env

    def init_record(self):
        # Record the simulation results
        self.revenue_record = None
        self.revenue_record = env_record(record_start_epoch=self.args.revenue_record_start,  # start record env revenue number
                                    averaged_stamp=self.args.revenue_averaged_stamp)

        # add agent plot
        self.revenue_record.init_agent_record(agent_name_list = self.agent_name_list)


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
        print('init global agent in mini signal game (Opening auction version )!!')
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
                args.agt_obs_public_signal_dim * agt_id) + '-> ' + str(args.agt_obs_public_signal_dim * (agt_id + 1) - 1))

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

    def set_agent_obs_dim(self,agent_name,observed_dim_list):

        self.agt_list[agent_name].set_public_signal_dim(observed_dim_list=observed_dim_list)



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

            new_agent = deepcopy(ascending_base_agent(args=args,agent_name=agt,
                                              obs_public_signal_dim=self.get_args_public_signal_dim(args,agent_name=agt),
                                              public_signal_range=args.public_signal_range,
                                              private_signal_range = args.private_signal_range,
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

        if self.init_private_signal :
            private_signal_generator = base_private_Signal_generator (
                dtype='Discrete', default_feature_dim=args.private_signal_dim,
                default_lower_bound=0,
                default_upper_bound=args.private_signal_range,
                default_generation_method='uniform',
                default_value_to_signal=0,
                agent_name_list=self.agent_name_list
            )
            self.private_signal_generator = private_signal_generator
            self.private_signal_generator.generate_default_iid_signal()



    def private_signal_to_value(self,agent_name,agt):
        private_value =0
        if self.private_signal_generator is not None:

            private_signal = self.get_agt_private_signal(agent_name=agent_name,agt=agt)
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

        if (not truncation):

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
            if self.private_signal_generator is None:
                print('not exist signal generator, skip signal updating ')
            return



        #receive the public signal results and saved

        self.public_signal_generator.generate_signal(data_type=data_type)  # update the signal

        self.last_public_signal = deepcopy(self.public_signal_generator.get_whole_signal_realization())
        #Full information generated from the platform

        return
    # private signal
    def update_agt_private_signal(self,agent_name):
        success = self.private_signal_generator.generate_agt_signal(agent_name=agent_name)

        return success
    def update_private_signal(self):

        for agent_name in self.agent_name_list:
            if self.check_private_signal_generator(agent_name=agent_name):

                succ = self.update_agt_private_signal(agent_name = agent_name)
                #print(succ)
        return
    def check_private_signal_generator(self,agent_name):
        if self.private_signal_generator is None or (not self.private_signal_generator.check_avaliable_agent_name(agent_name=agent_name)):
            return False
        else:
            return True
    def get_agt_private_signal(self,agent_name,agt):
        ## current using private signal generator
        if self.check_private_signal_generator(agent_name=agent_name) :

            private_signal = self.private_signal_generator.get_latest_signal(agent_name=agent_name)
        else:
            private_signal =None

        return private_signal



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

        for epoch in range(self.mini_env_iters):

            # begin open_ascending_auction with mini_env_iter times


            # update public signal before first agent move
            # platform movement
            self.update_env_public_signal()
            self.update_private_signal()

            # for each open auction, begin with an start
            for agent_name in env.agent_iter():
                # print(agent_name)

                observation, reward, termination, truncation, info = env.last()

                # env.render()  # this visualizes a single game
                _obs = observation['observation']
                agt = self.agt_list[agent_name]
                agt_idx = env.agents.index(agent_name)



                # get the true value from signal

                if (self.mini_item == 1 and (
                        (args.action_mode == 'same' and _obs == args.bidding_range + 1))) or (
                        self.mini_item > 1 and ((args.action_mode == 'div' and _obs == [
                    args.bidding_range * args.valuation_range + 1] * self.mini_item) or
                                                (args.action_mode == 'same' and _obs == [
                                                    args.bidding_range + 1] * self.mini_item))):
                    # the first round not compute reward
                    # In real scenario, signal realization/auction setup happens at the
                    # start of a auction, and auction allocation happens at the end. We
                    # simply the implementation by setting both action to the start by
                    # skipping them at the first round.

                    #print('first round skipping')

                    true_value = 0
                    info['highest_bid']=0  # assume highest_bid=0
                    info['tmp_allocation'] = 0                      # tmp_allocation=0
                    info['allocation'] = -1 # not finish

                    agt.re_init_tmp_action() # reset tmp_action history


                    pass

                elif termination :

                    # aucton is end


                    # Compute reward and update agent policy each round
                    allocation = info['allocation']
                    last_high_bid = info['highest_bid']

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
                        obs = increase_info_to_obs(obs, extra_info_name='true_value', value=true_value)
                        obs = increase_info_to_obs(obs, extra_info_name='payment', value=reward)

                    # update poicy
                    agt.update_policy(obs=obs, reward=final_reward, done=truncation)

                    # agt.record_reward(final_reward)
                    # record each step

                    if (self.current_step % 10000 ==0 ) :
                        # temp report
                        print('in the '+ str(self.current_step) + ' th auction : the winner is ' +
                              agent_name + '  final payment is ' + str(reward) + ' | his true value is ' + str(
                            true_value) + '| highest bid is ' + str(last_high_bid) + ' ｜ his final reward is '+ str(final_reward))


                    # self.add_reward_by_agt_name(agt_name=agent_name, reward=final_reward)

                    record_step(args, agt, allocation, agent_name,
                                self.current_step + self.mini_env_iters * self.current_round, env, reward,
                                self.revenue_record, agents_list=self.agent_name_list)

                # rebuild the observation

                public_signal = set_agent_signal_observation(
                    public_signal_generator=self.public_signal_generator,
                    agent_name=agent_name,
                    agt=agt,
                    mode='default'

                )

                obs = agt.receive_observation(args, self.budget, agent_name, agt_idx,
                                              extra_info=info,  #take use of the current bid | allocation |
                                              observation=observation,
                                              public_signal=public_signal,
                                              )

                # minimal raise as an extra info
                obs = increase_info_to_obs(obs, extra_info_name='minimal_raise', value=self.minimal_raise)

                # private information
                private_s = self.get_agt_private_signal(agent_name=agent_name,agt=agt)

                obs = increase_info_to_obs(obs,extra_info_name='private_signal',
                                               value=self.get_agt_private_signal(agent_name=agent_name,agt=agt)
                                               )


                if termination or truncation:
                    env.step(None)
                else:
                    # print(agent_name, 'agent_name')
                    last_high_bid = info['highest_bid']

                    # if temp winner : submit the former bid
                    if info['tmp_allocation']:
                        new_action = agt.get_last_tmp_action()
                        last_bid = agt.get_last_tmp_action()

                    else:
                        last_bid = agt.get_last_tmp_action()

                        new_action = agt.generate_action(obs)  # get the next round action based on the observed budget

                        new_action = self.check_ascending_price(new_action=new_action,last_bid=last_bid,last_high_bid=last_high_bid,tmp_allocation=False)


                    #agt.record_action(new_action)



                    env.step(new_action)

                    #print(str(agent_name) + ' obs=' + str(obs)+ 'new_action='  + str(new_action) + '| his last action is ' + str(last_bid) )

                    agt.record_tmp_action(new_action)

                # env.render()  # this visualizes a single game


            # end of 1 round open auction


            ## print setting
            self.current_step += 1

            self.print_results(env, args)

            self.reset_env() # reset open auction to re-init

        self.current_round += 1

        # print(self.mini_env_reward_list)
    def print_results(self,env,args,last_k_epoch=None):


            agents_list=self.agent_name_list
            agt_list=self.agt_list
            epoch = self.current_step + self.mini_env_iters * self.current_round
            prefix=None

            # add agent reward plot (only record after explore)
            #
            ignore_explore_flag = epoch > args.exploration_epoch

            if (epoch  % args.revenue_averaged_stamp == 0)  and ignore_explore_flag:
                #record all
                # last_k_epoch = plot the last k epoch reward
                self.record_all_agent_avg_reward(agent_list=agt_list, agent_name_list=agents_list, win_only=False, last_k_epoch=last_k_epoch)

                #record win only
                self.record_all_agent_avg_reward(agent_list=agt_list, agent_name_list=agents_list, win_only=True,
                                                 last_k_epoch=last_k_epoch)
            #

            if (epoch  % args.estimate_frequent == 0):
                estimations = {agent: [] for agent in agents_list}
                estimations = self.build_estimation(args, agent_list=agt_list, env_agent=agents_list,
                                                    estimation=estimations)

                self.plot_figure(path='./results', folder_name=str(args.exp_id), args=args,
                            epoch=int(epoch),
                            prefix=prefix,
                            estimations=estimations)

                self.revenue_record.plot_avg_revenue(
                    args, path='./results', folder_name=str(args.exp_id),
                    figure_name=get_figure_name(args, epoch=int(epoch)),
                    mechanism_name=get_mechanism_name(args),
                    # plot_y_range=[0.4*args.valuation_range,args.valuation_range] #plot range
                )

                if args.record_efficiency == 1:
                    self.revenue_record.plot_avg_efficiency(
                        args, path='./results', folder_name=str(args.exp_id),
                        figure_name=get_figure_name(args, epoch=int(epoch)),
                        mechanism_name=get_mechanism_name(args),
                        # plot_y_range=[0.4*args.valuation_range,args.valuation_range] #plot range
                    )

                #print agent reward
                # last_k_epoch = plot the last k epoch reward
                #print only win
                self.revenue_record.plot_agent_reward(
                    args,
                    plot_agent_list=agents_list,
                    avg=True,win_only=True,
                    last_k_epoch=last_k_epoch,
                    path='./results', folder_name=str(args.exp_id),
                    figure_name=get_figure_name(args, epoch=int(epoch)),
                    mechanism_name=get_mechanism_name(args),

                )
                #print all
                self.revenue_record.plot_agent_reward(
                    args,
                    plot_agent_list=agents_list,
                    avg=True, win_only=False,
                    last_k_epoch=last_k_epoch,
                    path='./results', folder_name=str(args.exp_id),
                    figure_name=get_figure_name(args, epoch=int(epoch)),
                    mechanism_name=get_mechanism_name(args),

                )


                # print revenue
                print_agent_reward(agent_list=agt_list, agent_name_list=agents_list,
                                   avg_epoch=args.revenue_averaged_stamp,
                                   filter_allocation=True
                                   )

                print('the latest avg revenue is ' + str(self.revenue_record.get_last_avg_revenue()))


    def record_all_agent_avg_reward(self,agent_list,agent_name_list,win_only=False,last_k_epoch=None):

        # only win  -> only win reward
        for agent_name in agent_name_list:
            agent = agent_list[agent_name]
            # get latest k reward
            if last_k_epoch is None:
                epoch = self.args.revenue_averaged_stamp
            else:
                epoch = last_k_epoch

            avg_reward  = agent.get_averaged_reward(epoch=epoch,allocated_only=win_only)

            #set the record
            self.revenue_record.record_agent_reward(agent_name, reward=avg_reward, avg=True, win_only=win_only)

    def max_signal_value(self):
        args=self.args

        #encode public signal
        highest_public_value =0
        if self.public_signal_generator is not None:

            each_sigal_value_range = args.public_signal_range  # [0-K]
            for j in range(args.agt_obs_public_signal_dim):
                highest_public_value += each_sigal_value_range * ((each_sigal_value_range + 1) ** j)

        # encode private signal
        highest_private_value =0
        if self.private_signal_generator is not None :
            private_signal_range = args.private_signal_range
            for j in range(args.private_signal_dim):
                highest_private_value += private_signal_range * ((private_signal_range + 1) ** j)

        if highest_private_value ==0:
            return highest_public_value
        if highest_public_value ==0 :
            return highest_private_value

        return highest_public_value * highest_private_value



    def build_estimation(self,args,agent_list,env_agent,estimation):
        signals = {agent: [] for agent in env_agent}

        for i, agent in enumerate(env_agent):
            # max_public_signal=[]*args.agt_obs_public_signal_dim
            # max_state =agent_list[agent].encode_signal(public_signal=,
            #                                            private_signal=None)
            if 'kl_sp' in args.folder_name and args.speicial_agt is not None and args.speicial_agt == agent:
                estimation = build_speicial_estimation(args, agent_list, agt_name=agent, estimation=estimation)
            else:
                max_state = self.max_signal_value()
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
                        #estimation[agent].append(action)
                        estimation[agent].append(action * 1.0 / partial_signal_value)

        return estimation

    def check_minimal_bid_increase(self,new_action=1,last_bid=0):
        # used for ascending bid check
        # at least bid higher than last bid
        # or direct give up bid ( = submit the last round bid)

        # if bid lower than last bid
        # then give up = submit last bid

        if new_action <= last_bid:
            return last_bid

        # if bid higher than last bid :
        # check minimal raise

        if new_action > last_bid +self.minimal_raise :
            return new_action
        else:
            return last_bid +self.minimal_raise

    def check_ascending_price(self,new_action=1,last_bid=0,last_high_bid=1,tmp_allocation=False):
        # based on the mechanism to adjust the last bid

        this_round_bid =0

        if tmp_allocation :
            # last round is winner
            # rule 0 : regard the last round winner not to submit
            return last_high_bid

        # rule 1 : have to be monotonically increasing
        if new_action < last_high_bid :
            # regard as to submit the last round bid
            this_round_bid = last_bid

        elif new_action > last_high_bid :
            # bid higher than the last winner bid :

            # rule 2 : at lease increase with minimal increase defined by the mechanism
            this_round_bid = self.check_minimal_bid_increase(new_action=new_action,last_bid=last_bid)
        else:
            # new_action == last_high_bid

            # some agent turn to submit the last win price as well
            this_round_bid = self.check_minimal_bid_increase(new_action=new_action,last_bid = last_high_bid)

        return this_round_bid


    def plot_figure(self,path, folder_name, args, epoch, prefix=None,
                estimations=[]):  # make extra folder named "first" |"second"
        import os
        import matplotlib.pyplot as plt
        save_dir = os.path.join(os.path.join(path, args.folder_name), folder_name)
        build_path(save_dir)

        figure_name = get_figure_name(args, epoch, prefix=prefix)
        saved_data = []
        # bid / value

        if args.public_signal and args.value_to_signal and \
                args.agt_obs_public_signal_dim == 1 and args.mechanism == 'second_price' \
                and args.public_bidding_range_asym == 0 and args.public_signal_asym == 0:
            index = range(len(estimations['player_0']))
            if args.player_num == 2:
                plot_data = [-np.log(0.5 * 0.1 * i) / 4 / (2 - 0.1 * i) * (4 * 0.1 * i) * 10 for i in index]
                plt.plot(index, plot_data, label="Optimal")
            if args.player_num == 3:
                plot_data = [20 * i / (20 + i) for i in index]
                plt.plot(index, plot_data, label="Optimal")
            if args.player_num > 3:
                plot_data = [optimal_bid(i, nplayer=args.player_num) for i in index[1:]]
                plt.plot(index[1:], plot_data, label="Optimal")

        for i in range(args.player_num):
            agt_name = 'player_' + str(i)

            plot_data = estimations[agt_name]

            saved_data.append(plot_data)
            plt.plot([x - 0.01 * i for x in range(len(plot_data))], plot_data, label=str(agt_name))

        if args.value_to_signal:
            plt.ylabel('bid')
        elif 'kl' in args.folder_name:
            plt.ylabel('bid/signal(x)')
        else:
            #plt.ylabel('bid')
            plt.ylabel('bid/valuation')
        if args.public_signal:
            plt.ylim(0, 4.0)
            plt.xlabel('partial signal(State)')
        else:
            plt.ylim(0, 2.0)
            plt.xlabel('valuation')
        plt.legend()
        print(os.path.join(save_dir, figure_name))
        plt.savefig(os.path.join(save_dir, figure_name))
        plt.show()
        plt.close()

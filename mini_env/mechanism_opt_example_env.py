from .simple_mini_env1 import *
from agent.normal_agent import *


class opt_examle_env(simple_mini_env1):
    def __init__(self, round, item_num, mini_env_iters, agent_num, args, policy,
                 set_env=None, init_from_args=True, winner_only=True, fixed_truthful=False
                 ):
        super(opt_examle_env, self).__init__(
            round, item_num, mini_env_iters, agent_num, args, policy,skip=False
        )

        self.winner_only = winner_only  # only winner knows the true value
        self.supported_function = ['CRRA', 'CARA']

        # global setting
        self.round = round  # K
        self.mini_item = item_num  # T
        self.agent_num = agent_num
        self.total_step = round * mini_env_iters  # total step = K*T
        self.args = args
        self.fixed_truthful = fixed_truthful

        self.agent_name_list = []  # global agent name list

        self.custom_env = set_env

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

        if init_from_args:
            self.init_mini_env_setting_from_args()
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

            self.agt_list = {agent: None for agent in self.custom_env.agents}

            # init reward list
            self.init_reward_list()

        self.init_global_rl_agent()
        self.assign_generate_true_value(self.agent_name_list)

        self.init_record()

    # use truthful agent
    def change_to_truthful(self):
        self.fixed_truthful = True
        self.reset_env()

    def change_to_greedy(self):
        self.fixed_truthful = False
        self.reset_env()

    def re_init(self,init_from_args=True):
        if init_from_args:
            self.init_mini_env_setting_from_args()
            self.init_custom_env()

        self.current_step=0
        # may be none

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
        print('reinit the whole envs including agents ')


    def reset_env(self):

        self.current_step = 0
        self.custom_env.reset()
        self.mini_agent_name_list = self.custom_env.agents
        self.mini_policy.assign_agent(
            self.mini_agent_name_list)  # assign the list of agent into the allocation and payment rule

        # self.init_record()
        # self.init_reward_list()
        # self.init_global_rl_agent()
        #
        # self.assign_generate_true_value(self.agent_name_list)

    # change the  agent
    def init_global_rl_agent(self):
        if self.agt_list is not None:
            highest_value = self.args.valuation_range - 1
            for i, agt in enumerate(self.custom_env.agents):

                if self.fixed_truthful:
                    new_agent = deepcopy(fixed_agent(
                        agent_name=agt, highest_value=highest_value))
                else:
                    new_agent = deepcopy(simple_greedy_agent(
                        args=self.args, agent_name=agt, highest_value=highest_value))

                self.agt_list[agt] = new_agent

        else:
            print('agent list is None, init RL agents failed')

    # assign the true value generator
    def assign_generate_true_value(self, agent_name_list):

        def generate_true_value(self):
            # use the uniform distribution
            true_value = random.randint(self.lowest_value, self.highest_value)

            self.record_true_value(true_value)

        for name in agent_name_list:
            bind(self.agt_list[name], generate_true_value,print_flag=False)
            #print(f'bind (function) generate_true_value for agent {name}')

    # here step means to converge to the equilibrium

    def step(self):
        if self.current_round % self.args.estimate_frequent == 0:
            print('round is ' + str(self.current_round))
            # begin K round in single round

        env = self.custom_env
        self.current_step = 0
        args = self.args


        public_signal = None

        for agent_name in env.agent_iter():
            # print(agent_name)

            observation, reward, termination, truncation, info = env.last()

            # env.render()  # this visualizes a single game
            _obs = observation['observation']
            agt = self.agt_list[agent_name]
            agt_idx = env.agents.index(agent_name)

            if (not termination) and (not truncation):
                agt.generate_true_value()
                # print(agt.get_latest_true_value())

            if (self.mini_item == 1 and (
                    (args.action_mode == 'div' and _obs == args.bidding_range * args.valuation_range + 1) or
                    (args.action_mode == 'same' and _obs == args.bidding_range + 1))) or \
                    (self.mini_item > 1 and ((args.action_mode == 'div' and _obs == [
                        args.bidding_range * args.valuation_range + 1] * self.mini_item) or
                                             (args.action_mode == 'same' and _obs == [
                                                 args.bidding_range + 1] * self.mini_item))):
                # the first round not compute reward
                # In real scenario, signal realization/auction setup happens at the
                # start of a auction, and auction allocation happens at the end. We
                # simply the implementation by setting both action to the start by
                # skipping them at the first round.
                #print('first round skipping')
                allocation = None  # added fixed bug
                pass

            else:  # Compute reward and update agent policy each round
                allocation = info['allocation']
                # get true value for computing the reward

                # obs record
                obs = agt.get_last_obs()

                if (not termination) and (not truncation):
                    true_value = agt.get_last_round_true_value()
                else:  # done, not generate new value
                    true_value = agt.get_latest_true_value()

                # adjust each agent with his reward function
                final_reward = compute_reward(true_value=true_value, pay=reward,
                                              allocation=allocation,
                                              reward_shaping=check_support_reward_shaping(self.supported_function,
                                                                                          args.reward_shaping),
                                              reward_function_config=build_reward_function_config(args),
                                              user_id=agent_name, budget=self.budget,
                                              args=args, info=info  # compute based on the former budget
                                              )  # reward shaping

                last_action = agt.get_latest_action()

                # update poicy
                agt.update_policy(obs=obs, reward=final_reward, done=truncation)
                agt.record_reward(final_reward)
                # record each step

                self.add_reward_by_agt_name(agt_name=agent_name, reward=final_reward)
                record_step(args, agt, allocation, agent_name,
                            self.current_step + self.mini_env_iters * self.current_round, env, reward,
                            self.revenue_record, agents_list=self.agent_name_list)

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

            next_true_value = agt.get_latest_true_value()

            if termination or truncation:
                env.step(None)
            else:

                new_action = agt.generate_action(obs)  # get the next round action based on the observed budget
                agt.record_action(new_action)

                ## cooperate infomation processing [eric]
                submit_info_to_env(args, agt_idx, self.mini_policy, agent_name, next_true_value)

                env.step(new_action)
            ## print setting
            self.current_step += 1

            self.print_results(env, agent_name,args)

            # env.render()  # this visualizes a single game

        self.current_round += 1

        return self.revenue_record.get_last_avg_revenue()  # latest platform reward

    def init_output_dataset(self):
        # can rewrite dataset formate here

        dataset = {agent: {} for agent in self.agent_name_list}
        for agent in self.agent_name_list:
            dataset[agent]['true_value']=[]
            dataset[agent]['bidding'] = []
            dataset[agent]['reward'] = []
            dataset[agent]['payment'] = []
            dataset[agent]['truthful_bidding_reward'] = []



        return dataset


    def format_output_data(self,dataset,agent_name,true_value,bidding,reward,payment,truthful_bidding_reward=None):

        if agent_name in dataset:

            if true_value is not None:
                dataset[agent_name]['true_value'].append(true_value)
            if bidding is not None:
                dataset[agent_name]['bidding'].append(bidding)
            if reward is not None:
                dataset[agent_name]['reward'].append(reward)
            if payment is not None:
                dataset[agent_name]['payment'].append(payment)

            if truthful_bidding_reward is not None:
                dataset[agent_name]['truthful_bidding_reward'].append(truthful_bidding_reward)




        return dataset

    # use this method to generate data set and reward for the mechanism training
    def generate_dataset(self, data_num=10, compare_truthful=None):
        # after converge test the results and build dataset

        # here we define compare truthful  = compute the truthful bidding results + if each agent not truthful under others truthful regret

        if compare_truthful is not None and compare_truthful:
            self.mini_policy.close_grad()
        else:
            self.mini_policy.open_grad()


        # init
        self.reset_env()
        args = self.args

        dataset=self.init_output_dataset()

        # compare with truthful
        for i in range(data_num+1):
            env = self.custom_env
            for agent_name in self.agent_name_list:
                # generate with
                observation, reward, termination, truncation, info = env.last()

                # env.render()  # this visualizes a single game
                _obs = observation['observation']
                agt = self.agt_list[agent_name]
                agt_idx = env.agents.index(agent_name)

                if (not termination) and (not truncation):
                    agt.generate_true_value()
                    # print(agt.get_latest_true_value())

                if args.action_mode == 'same' and _obs == args.bidding_range + 1:
                    allocation = None  # added fixed bug
                    pass

                else:  # Compute reward and update agent policy each round
                    allocation = info['allocation']
                    # get true value for computing the reward

                    # obs record
                    obs = agt.get_last_obs()

                    if (not termination) and (not truncation):
                        true_value = agt.get_last_round_true_value()
                    else:  # done, not generate new value
                        true_value = agt.get_latest_true_value()

                    # adjust each agent with his reward function
                    final_reward = compute_reward(true_value=true_value, pay=reward,
                                                  allocation=allocation,
                                                  reward_shaping=check_support_reward_shaping(self.supported_function,
                                                                                              args.reward_shaping),
                                                  reward_function_config=build_reward_function_config(args),
                                                  user_id=agent_name, budget=self.budget,
                                                  args=args, info=info  # compute based on the former budget
                                                  )  # reward shaping

                    # compute the expected reward after convergence on each agent

                    if compare_truthful is not None and compare_truthful:
                        final_reward=None


                    dataset=self.format_output_data(dataset,agent_name,true_value,bidding=agt.get_latest_action(),
                                                        reward=final_reward,payment=reward
                                                        )






                #next step
                obs = agt.receive_observation(args, self.budget, agent_name, agt_idx,
                                              extra_info=None, observation=observation,
                                              public_signal=None,
                                              true_value_list=agt.get_true_value_history(),
                                              action_history=agt.get_action_history(),
                                              reward_history=agt.get_reward_history(),
                                              allocation_history=agt.get_allocation_history()
                                              )
                if not self.winner_only or allocation:
                    # add the information of the payment into the obs
                    obs = increase_info_to_obs(obs, extra_info_name='payment', value=reward)

                agt.record_obs(obs)

                next_true_value = agt.get_latest_true_value()

                if termination or truncation:
                    env.step(None)
                else:

                    new_action = agt.generate_action(obs)  # get the next round action based on the observed budget
                    agt.record_action(new_action)

                    ## cooperate infomation processing [eric]
                    submit_info_to_env(args, agt_idx, self.mini_policy, agent_name, next_true_value)

                    env.step(new_action)
                ## print setting
                self.current_step += 1



        if compare_truthful is not None and compare_truthful:
            # compute the reward divergence between truthful bidding and greedy bidding
            # repeat bidding and compute different results
            self.mini_policy.open_grad()

            # reinitial the env that compute the expected ex post regret
            for i in range(data_num):
                self.custom_env.reset()
                env = self.custom_env

                # first compute truthful bidding reward
                for agent_name in self.agent_name_list:
                    true_value = dataset[agent_name]['true_value'][i]
                    env.step(true_value)



                # then compute regret the if one not truthful bidding

                for j in range(len(self.agent_name_list) + 1):
                    # select which agent is not truthful bidding while other is
                    if j ==len(self.agent_name_list):
                        greedy_agent_name=None
                        greedy_bid=None
                    else:
                        greedy_agent_name = self.agent_name_list[j]
                        greedy_bid = dataset[greedy_agent_name]['bidding'][i]

                    for k in range(len((self.agent_name_list))):

                        agent_name = self.agent_name_list[k]

                        observation, reward, termination, truncation, info = env.last()
                        allocation = info['allocation']
                        true_value=dataset[agent_name]['true_value'][i]


                        # adjust each agent with his reward function
                        final_reward = compute_reward(true_value=true_value, pay=reward,
                                                      allocation=allocation,
                                                      reward_shaping=check_support_reward_shaping(supported_function,
                                                                                                  args.reward_shaping),
                                                      reward_function_config=build_reward_function_config(args),
                                                      user_id=agent_name, budget=self.budget,
                                                      args=args, info=info  # compute based on the former budget
                                                      )  # reward shaping

                        if j == 0: # the first is the truthful results
                            dataset[agent_name]['truthful_bidding_reward'].append(final_reward)

                        elif j-1==k: #last round non truthful bid results | eg round 1 is the agent0 submit non-truthful
                            dataset[agent_name]['reward'].append(final_reward)

                        if j ==k : # non_truthful
                            env.step(greedy_bid)
                        else:
                            env.step(true_value)





        return dataset

    def print_results(self, env, agent_name,args, last_k_epoch=None):

        agents_list = self.agent_name_list
        agt_list = self.agt_list
        epoch = self.current_step #+ self.mini_env_iters * self.current_round
        prefix = str(self.round)

        # add agent reward plot (only record after explore)
        #
        ignore_explore_flag = (epoch / args.player_num) > args.exploration_epoch

        if ((epoch / args.player_num) % args.revenue_averaged_stamp == 0) and (
                agent_name == agents_list[-1]) and ignore_explore_flag:
            # record all
            # last_k_epoch = plot the last k epoch reward
            self.record_all_agent_avg_reward(agent_list=agt_list, agent_name_list=agents_list, win_only=False,
                                             last_k_epoch=last_k_epoch)

            # record win only
            self.record_all_agent_avg_reward(agent_list=agt_list, agent_name_list=agents_list, win_only=True,
                                             last_k_epoch=last_k_epoch)
        #

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

            # print agent reward
            # last_k_epoch = plot the last k epoch reward
            # print only win
            self.revenue_record.plot_agent_reward(
                args,
                plot_agent_list=agents_list,
                avg=True, win_only=True,
                last_k_epoch=last_k_epoch,
                path='./results', folder_name=str(args.exp_id),
                figure_name=get_figure_name(args, epoch=int(epoch / args.player_num)),
                mechanism_name=get_mechanism_name(args),

            )
            # print all
            self.revenue_record.plot_agent_reward(
                args,
                plot_agent_list=agents_list,
                avg=True, win_only=False,
                last_k_epoch=last_k_epoch,
                path='./results', folder_name=str(args.exp_id),
                figure_name=get_figure_name(args, epoch=int(epoch / args.player_num)),
                mechanism_name=get_mechanism_name(args),

            )

            # print revenue
            print_agent_reward(agent_list=agt_list, agent_name_list=agents_list,
                               avg_epoch=args.revenue_averaged_stamp,
                               filter_allocation=True
                               )

            print('the latest avg revenue is ' + str(self.revenue_record.get_last_avg_revenue()))

    def record_all_agent_avg_reward(self, agent_list, agent_name_list, win_only=False, last_k_epoch=None):

            # only win  -> only win reward
            for agent_name in agent_name_list:
                agent = agent_list[agent_name]
                # get latest k reward
                if last_k_epoch is None:
                    epoch = self.args.revenue_averaged_stamp
                else:
                    epoch = last_k_epoch

                avg_reward = agent.get_averaged_reward(epoch=epoch, allocated_only=win_only)

                # set the record
                self.revenue_record.record_agent_reward(agent_name, reward=avg_reward, avg=True, win_only=win_only)

    def build_estimation(self, args, agent_list, env_agent, estimation):

            for i, agent in enumerate(env_agent):

                highest_value = args.valuation_range
                # from the function ->init_global_rl_agent()

                for true_value in range(highest_value):
                    action = agent_list[agent].test_policy(true_value)

                    if true_value == 0:

                        estimation[agent].append(action)
                    else:

                        estimation[agent].append(action * 1.0 / true_value)

            return estimation

    def plot_figure(self, path, folder_name, args, epoch, prefix=None,
                        estimations=[]):  # make extra folder named "first" |"second"
            import os
            import matplotlib.pyplot as plt
            save_dir = os.path.join(os.path.join(path, args.folder_name), folder_name)
            build_path(save_dir)

            figure_name = get_figure_name(args, epoch, prefix=prefix)
            saved_data = []
            # bid / value

            for i in range(args.player_num):
                agt_name = 'player_' + str(i)

                plot_data = estimations[agt_name]

                saved_data.append(plot_data)
                plt.plot([x - 0.01 * i for x in range(len(plot_data))], plot_data, label=str(agt_name))

            plt.ylabel('bid/valuation')

            plt.ylim(0, 2.0)
            plt.xlabel('valuation')

            plt.legend()
            print(os.path.join(save_dir, figure_name))
            plt.savefig(os.path.join(save_dir, figure_name))
            plt.show()
            plt.close()

from .simple_mini_env1 import *
from agent.normal_agent import *
from Stackelberg_Game.stackelberg_agent_utils import stgame_greedy_agent


class stackelberg_game(simple_mini_env1):
    def __init__(self, round, item_num, mini_env_iters, agent_num, args, policy,
                 set_env=None, init_from_args=True, winner_only=True, fixed_truthful=False,
                 leading_agent_id=0,reward_mode ='revenue',leading_agent_update_flag=True
                 ):
        super(stackelberg_game, self).__init__(
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

        # stackelberg_setting
        self.leading_agent_id=leading_agent_id
        self.leading_agent_name = self.agent_name_list[leading_agent_id]
        self.reward_mode=reward_mode

        self.leading_agent_update_flag=leading_agent_update_flag
        self.follower_agent_update_flag= (leading_agent_update_flag)

        #follower not update policy
        self.set_follower_agent_list(leading_agent_name_list=[self.leading_agent_name])

        # discount and information enable
        self.requires_discount = False
        self.requires_information = False
        self.leader_discount_update_flag = True
        self.leader_info_update_flag = False

        self.shared_info = None

        self._running_full_stackelberg = False

        self.init_truthful_info_flag = True

    def init_truthful_info(self):
        self.init_truthful_info_flag = True

    def run_full_stackelberg(self):
        self._running_full_stackelberg = True

    def enable_discount(self):
        self.requires_discount = True

    def enable_information(self):
        self.requires_information = True

    def disable_discount(self):
        self.requires_discount = False

    def disable_information(self):
        self.requires_information = False

    def turnon_leader_discount_update(self):
        self.leader_discount_update_flag = True

    def turnoff_leader_discount_update(self):
        self.leader_discount_update_flag = False

    def turnon_leader_info_update(self):
        self.leader_info_update_flag = True

    def turnoff_leader_info_update(self):
        self.leader_info_update_flag = False

    # update agent control
    def turnon_leading_agent_update(self):
        self.leading_agent_update_flag=True

    def turnoff_leading_agent_update(self):
        self.leading_agent_update_flag = False

    def turnon_follower_agent_update(self):
        self.follower_agent_update_flag = True

    def turnoff_follower_agent_update(self):
            self.follower_agent_update_flag = False
    # agent algo export

    def export_agent(self,agent_name):
        if agent_name in self.agent_name_list:
            # use deep copy to export
            # notice to see whether all the algo with agent is export
            return deepcopy(self.agt_list[agent_name])
    def export_all_agent(self):
        all_agents = {}
        for agent_name in self.agent_name_list:
            all_agents[agent_name] = self.export_agent(agent_name)
        return all_agents
    #
    def set_follower_agent_list(self,leading_agent_name_list=[]):
        temp_list=[]
        for agent_name in self.agent_name_list:
            if len(leading_agent_name_list)==0 or (agent_name not in leading_agent_name_list):
                temp_list.append(agent_name)

        self.follower_agent_list=temp_list



    def adjust_reward_mode(self,reward_mode=None):
        if reward_mode is not None:
            self.reward_mode=reward_mode
    def get_reward_mode(self):
        return self.reward_mode


    def customed_reward(self,agent_name,cost=0,bid=3,allocation=0,mode='revenue',other_data=[]):

        if agent_name == self.leading_agent_name : # or in self.adjust reward
            #special reward mode for leading agent
            if mode =='income':
                return bid * allocation # compute the income
            elif mode =='revenue_percentage':
                #require future work
                current_revenue = other_data['total_revenue'][agent_name]
                return (bid - cost)*allocation+current_revenue / ( sum(other_data['total_revenue']) + (bid - cost)*allocation)
            elif mode =='revenue':
                return (bid - cost) * allocation

        else: #other agent
            # please check cost and bid is >0 or <0
            return (bid - cost)*allocation

    # return agent policy

    # use truthful agent
    def change_to_truthful(self):
        self.fixed_truthful = True
        self.reset_env()

    def change_to_greedy(self):
        self.fixed_truthful = False
        self.reset_env()

    def re_init(self,init_from_args=True,leading_agent_id=None,leading_agent_update_flag=None):
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

        # adjust leading agent id
        if leading_agent_id is not None:
            self.leading_agent_id = leading_agent_id
        self.leading_agent_name = self.agent_name_list[self.leading_agent_id]
        self.set_follower_agent_list(leading_agent_name_list=[self.leading_agent_name])

        if leading_agent_update_flag is not None:
            self.leading_agent_update_flag=leading_agent_update_flag
            self.follower_agent_update_flag= (not leading_agent_update_flag)
        else:
            self.leading_agent_update_flag =True
            self.follower_agent_update_flag =True


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
            # action with info:
            highest_value = self.args.valuation_range - 1
            for i, agt in enumerate(self.custom_env.agents):

                if self.fixed_truthful:
                    new_agent = deepcopy(fixed_agent(
                        agent_name=agt, highest_value=highest_value))
                else:
                    new_agent = deepcopy(stgame_greedy_agent(
                        args=self.args, agent_name=agt, highest_value=highest_value))

                self.agt_list[agt] = new_agent

        else:
            print('agent list is None, init RL agents failed')

    # assign the true value generator
    def assign_generate_true_value(self, agent_name_list):

        def generate_true_value(self):
            # use the uniform distribution
            true_value = random.randint(self.lowest_value, self.highest_value)
            #print('generate use uniform distribution !')
            self.record_true_value(true_value)

        for name in agent_name_list:
            bind(self.agt_list[name], generate_true_value,print_flag=False)
            #print(f'bind (function) generate_true_value for agent {name}')

    # generate shaered info(if there is)
    def generate_shared_info(self):
        if self.requires_information:
            leading_agent = self.agt_list[self.leading_agent_name]
            shared_info = leading_agent.share_info(truthful_reveal=self.init_truthful_info_flag)

            if shared_info is None:
                raise ValueError("shared information should not be None!")
            self.shared_info = shared_info
        else:
            self.shared_info = 0

    # here step means to converge to the equilibrium

    def step(self):
        if self.current_round % self.args.estimate_frequent == 0:
            print('round is ' + str(self.current_round))
            # begin K round in single round

        env = self.custom_env
        self.current_step = 0
        args = self.args

        # set not truthful, if info algorithm(info update) is used
        if self.requires_information and self.leader_info_update_flag:
            self.init_truthful_info_flag = False

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

                # once value is generated, leading agent will share info about discount
                if agent_name == 'player_0':
                    # generate shared information at loop start
                    self.generate_shared_info()

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
                # final_reward = compute_reward(true_value=true_value, pay=reward,
                #                               allocation=allocation,
                #                               reward_shaping=check_support_reward_shaping(self.supported_function,
                #                                                                           args.reward_shaping),
                #                               reward_function_config=build_reward_function_config(args),
                #                               user_id=agent_name, budget=self.budget,
                #                               args=args, info=info  # compute based on the former budget
                #                               )  # reward shaping

                last_action = agt.get_latest_action()
                # adjust the stackelberg mode
                final_reward = self.customed_reward(
                    agent_name=agent_name,
                    cost = true_value,
                    bid=reward, # or the last_action (should be equal )
                    allocation=allocation,
                    mode =self.reward_mode

                )

                income = reward * allocation
                revenue = (reward-true_value) * allocation
                agt.record_all_type_reward(revenue=revenue,income=income)

                agt.record_reward(final_reward)


                last_action = agt.get_latest_action()

                # update poicy
                if agent_name in self.follower_agent_list:
                    # if follower
                    if self.follower_agent_update_flag:
                        # update poicy
                        agt.update_policy(obs=obs, reward=final_reward, done=truncation)
                else:
                    #if leader
                    if self.leading_agent_update_flag:
                        # update poicy
                        agt.update_policy(obs=obs, reward=final_reward, done=truncation)
                    # update leader's discount func if discount is required
                    if self.requires_discount and self.leader_discount_update_flag:
                        # update discount policy
                        agt.update_discount(obs=obs, reward=final_reward, done=truncation)
                    if self.requires_information and self.leader_info_update_flag:
                        # update info signal policy
                        agt.update_info(obs=obs,reward=final_reward,done=truncation)


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


            if agent_name != self.leading_agent_name:
                obs = increase_info_to_obs(obs, extra_info_name='leader_info', value=self.shared_info)
            else:
                obs = increase_info_to_obs(obs,extra_info_name='leader_info', value=0)
            # if agent_name ==self.leading_agent_name:
            #     # leader decide to share information
            #     shared_info = agt.share_info() #output = info or None
            # obs = increase_info_to_obs(obs,extra_info_name='leader_info',value=shared_info)


            agt.record_obs(obs)

            next_true_value = agt.get_latest_true_value()

            if termination or truncation:
                env.step(None)
            else:

                new_action = agt.generate_action(obs)  # get the next round action based on the observed budget
                agt.record_action(new_action)

                submit_action = new_action
                # if requires discount, generate discount value for leading agent
                if self.requires_discount:
                    if agt.agent_name == self.leading_agent_name:
                        discount = agt.generate_discount(obs)
                        submit_action = submit_action-discount
                        if submit_action < 0:
                            submit_action = 0

                ## cooperate infomation processing [eric]
                submit_info_to_env(args, agt_idx, self.mini_policy, agent_name, next_true_value)

                env.step(submit_action)
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
                                                      reward_shaping=check_support_reward_shaping(['CRRA','CARA'],
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

        # saving estimation results, do not save for stackelberg game (too many results)
        if ((epoch / args.player_num) % args.estimate_frequent == 0) and (agent_name == agents_list[-1]) and (not self._running_full_stackelberg):
            print('saving estimation results...')
            if self.requires_information:
                # for leading agent: true_value -> act-discount
                # for follower agent: true_value + info -> act
                # get possible infos
                leader_agt = self.agt_list[self.leading_agent_name]
                possible_discounts = set([leader_agt.test_discount(true_value) for true_value in range(self.args.valuation_range)])
                possible_infos = [leader_agt.test_info(discount,self.init_truthful_info_flag) for discount in possible_discounts]
                for agt_name in self.agent_name_list:
                    if agt_name == self.leading_agent_name:
                        estimation = self.build_estimation_info(agt_name,is_leader=True,possible_infos=[0])
                    else:
                        estimation = self.build_estimation_info(agt_name,is_leader=False,possible_infos=possible_infos)
                    # plot for single agent
                    self.plot_figure(path='./results', folder_name=str(args.exp_id) + self.get_exp_post_fix(),
                                figure_name = agt_name + '_policy_epoch_{}'.format(int(epoch / args.player_num)),
                                args=args,
                                epoch=int(epoch / args.player_num),
                                estimations=estimation)
            else:
                estimations = {agent: [] for agent in agents_list}

                estimations = self.build_estimation(args, agent_list=agt_list, env_agent=agents_list,
                                                    estimation=estimations)

                self.plot_figure(path='./results', folder_name=str(args.exp_id) + self.get_exp_post_fix(), args=args,
                            epoch=int(epoch / args.player_num),
                            estimations=estimations)

            self.revenue_record.plot_avg_revenue(
                args, path='./results', folder_name=str(args.exp_id) + self.get_exp_post_fix(),
                figure_name=get_figure_name(args, epoch=int(epoch / args.player_num)),
                mechanism_name=get_mechanism_name(args),
                # plot_y_range=[0.4*args.valuation_range,args.valuation_range] #plot range
            )

            if args.record_efficiency == 1:
                self.revenue_record.plot_avg_efficiency(
                    args, path='./results', folder_name=str(args.exp_id) + self.get_exp_post_fix(),
                    figure_name=get_figure_name(args, epoch=int(epoch / args.player_num)),
                    mechanism_name=get_mechanism_name(args),
                    # plot_y_range=[0.4*args.valuation_range,args.valuation_range] #plot range
                )

            # print agent reward
            # last_k_epoch = plot the last k epoch reward
            # print only win
            # self.revenue_record.plot_agent_reward(
            #     args,
            #     plot_agent_list=agents_list,
            #     avg=True, win_only=True,
            #     last_k_epoch=last_k_epoch,
            #     path='./results', folder_name=str(args.exp_id) + self.get_exp_post_fix(),
            #     figure_name=get_figure_name(args, epoch=int(epoch / args.player_num)),
            #     mechanism_name=get_mechanism_name(args),
            #
            # )
            # print total reward
            # print leader's reward
            leader_list = [self.leading_agent_name]
            self.revenue_record.plot_agent_reward(
                args,
                plot_agent_list=leader_list,
                avg=True, win_only=False,
                last_k_epoch=last_k_epoch,
                path='./results', folder_name=str(args.exp_id) + self.get_exp_post_fix(),
                figure_name='leader_{}_'.format(self.reward_mode) + get_figure_name(args, epoch=int(epoch / args.player_num)),
                mechanism_name=get_mechanism_name(args),

            )
            # print follower's reward
            follower_list = deepcopy(agents_list)
            follower_list.remove(self.leading_agent_name)
            self.revenue_record.plot_agent_reward(
                args,
                plot_agent_list=follower_list,
                avg=True, win_only=False,
                last_k_epoch=last_k_epoch,
                path='./results', folder_name=str(args.exp_id) + self.get_exp_post_fix(),
                figure_name='follower_'+get_figure_name(args, epoch=int(epoch / args.player_num)),
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


    def build_estimation_info(self, agent_name, is_leader=False, possible_infos=[]):
        highest_value = self.args.valuation_range
        estimation = {}
        for info in possible_infos:
            if is_leader:
                record_name = agent_name
            else:
                record_name = agent_name + '_info_' +str(info)
            estimation[record_name] = []
            for true_value in range(highest_value):
                if is_leader:
                    action = self.agt_list[agent_name].test_with_discount(true_value)
                else:
                    action = self.agt_list[agent_name].test_policy(true_value, leader_info=info)

                estimation[record_name].append(action)

        return estimation


    def build_estimation(self, args, agent_list, env_agent, estimation):

            for i, agent in enumerate(env_agent):

                highest_value = args.valuation_range
                # from the function ->init_global_rl_agent()

                for true_value in range(highest_value):
                    if self.requires_discount and agent == self.leading_agent_name:
                        action = agent_list[agent].test_with_discount(true_value)
                    else:
                        action = agent_list[agent].test_policy(true_value)

                    estimation[agent].append(action)

            return estimation

    def get_exp_post_fix(self):
        is_discount = '_discount' if self.requires_discount else '_no_discount'
        is_info = '_info' if self.requires_information else '_no_info'
        follower_update = '_follow_update' if self.follower_agent_update_flag else '_no_follow_update'
        reward_mode = '_{}'.format(self.reward_mode)
        discount_update = '_discount_update' if self.leader_discount_update_flag else '_no_discount_update'
        info_update = '_info_update' if self.leader_info_update_flag else '_no_info_update'
        postfix =  follower_update + is_discount + is_info + reward_mode + discount_update + info_update
        # if self._running_full_stackelberg:
        #     postfix = '_stkbg' + postfix
        return  postfix

    def plot_figure(self, path, folder_name, args, epoch, figure_name=None,
                        estimations={}, is_info=False):  # make extra folder named "first" |"second"
            import os
            import matplotlib.pyplot as plt
            save_dir = os.path.join(os.path.join(path, args.folder_name), folder_name)
            build_path(save_dir)

            if figure_name is None:
                figure_name = get_figure_name(args, epoch)

            # bid / value
            i = 0
            for record_name in estimations:
                plot_data = estimations[record_name]
                plt.plot([x - 0.01 * i for x in range(len(plot_data))], plot_data, label=record_name)
                i+=1

            if is_info:
                plt.ylabel('info')
            else:
                plt.ylabel('bid')
            plt.xlabel('cost')

            plt.legend()
            print(os.path.join(save_dir, figure_name))
            plt.savefig(os.path.join(save_dir, figure_name))
            plt.show()
            plt.close()

from agent.custom_agent import *


def get_discount_algorithm_for_leading_agent(leader_agt, args):
    leader_agt.discount_algorithm = EpsilonGreedy_auction(bandit_n=args.valuation_range * args.discount_range,
                                                          bidding_range=args.discount_range, eps=0.01,
                                                          start_point=int(args.exploration_epoch),
                                                          overbid=args.overbid, step_floor=int(args.step_floor),
                                                          signal=args.public_signal)


def get_info_algorithm_for_leading_agent(leader_agt, args, init=True):
    leader_agt.info_algorithm = EpsilonGreedy_auction(bandit_n=args.discount_range * args.info_algo_range,
                                                      bidding_range=args.info_algo_range, eps=0.01,
                                                      start_point=int(args.exploration_epoch),
                                                      overbid=args.overbid, step_floor=int(args.step_floor),
                                                      signal=args.public_signal)

def reset_algos_for_trained_agents(trained_agents,args,leading_agent_name,epoch):
    if epoch % 2 == 0:
        # leader
        if epoch % 4 == 0:
            # discount:
            reset_discount_algorithm_for_leading_agent(trained_agents[leading_agent_name],args)
        else:
            # info
            reset_info_algorithm_for_leading_agent(trained_agents[leading_agent_name],args)
    else:
        # follower
        for agent_name,agt in trained_agents.items():
            if agent_name != leading_agent_name:
                agt.reset_algo()

def reset_info_algorithm_for_leading_agent(leader_agt, args):
    get_info_algorithm_for_leading_agent(leader_agt, args, init=False)


def reset_discount_algorithm_for_leading_agent(leader_agt, args):
    get_discount_algorithm_for_leading_agent(leader_agt, args)


def generate_discount_for_leading_agent(self, obs):
    if hasattr(self, 'prefetched_discount'):
        # if agt has revealed discount, just use it
        discount_act = self.prefetched_discount
    else:
        true_value = self.get_latest_true_value()
        encoded_discount = self.discount_algorithm.generate_action(obs=true_value)
        discount_act = encoded_discount % self.discount_algorithm.bidding_range

        # action - (discount + 1) should not less than cost
        decoded_action = self.action_history[-1]
        if decoded_action - (discount_act + 1) < true_value:
            discount_act = max(decoded_action - true_value, 1) - 1

    if not hasattr(self, 'discount_history'):
        self.discount_history = []

    self.discount_history.append(discount_act)

    # plus 1: discount should start with 1
    return discount_act + 1


def update_discount_for_leading_agent(self, obs, reward, done=False):
    last_discount = self.discount_history[-1]
    if done:
        true_value = self.get_latest_true_value()
    else:
        true_value = self.get_last_round_true_value()

    encoded_discount = last_discount + true_value * (self.discount_algorithm.bidding_range)
    self.discount_algorithm.update_policy(encoded_discount, reward=reward)

    return


def test_policy_with_discount(self, true_value):
    # action policy: value * signal -> act
    encoded_value = true_value * self.args.info_signal_range + 0    # no info signal=0
    encoded_action = self.algorithm.generate_action(obs=encoded_value, test=True)
    action = encoded_action % (self.algorithm.bidding_range)

    # discount policy: value -> discount
    encoded_discount = self.discount_algorithm.generate_action(obs=true_value, test=True)
    discount = encoded_discount % (self.discount_algorithm.bidding_range)

    return action - (discount + 1)


def update_info_for_leading_agent(self, obs, reward, done=False):
    if done:
        last_info = self.info_history[-1]
    else:
        last_info = self.info_history[-2]

    last_discount = self.discount_history[-1]

    encoded_info = last_info + last_discount * self.info_algorithm.bidding_range
    self.info_algorithm.update_policy(encoded_info, reward=reward)


def test_info_policy(self, discount, truthful=False):
    # info policy: discount -> info
    if truthful:
        info = discount
    else:
        encoded_info = self.info_algorithm.generate_action(obs=discount, test=True)
        info = encoded_info % self.info_algorithm.bidding_range

    # info starts from 1
    return info + 1


def test_discount_policy(self, true_value):
    # discount policy: cost -> info
    encoded_discount = self.discount_algorithm.generate_action(obs=true_value, test=True)
    discount = encoded_discount % self.discount_algorithm.bidding_range

    return discount


def share_info_with_discount(self, truthful_reveal=True):
    true_value = self.get_latest_true_value()
    # generate bid action
    encoded_value = true_value * self.args.info_signal_range + 0    # no info signal=0
    encoded_action = self.algorithm.generate_action(obs=encoded_value)
    decoded_action = encoded_action % (self.algorithm.bidding_range)

    # generate discount with cost|bid
    encoded_discount = self.discount_algorithm.generate_action(obs=true_value)
    discount_act = encoded_discount % (self.discount_algorithm.bidding_range)

    # action - (discount + 1) should not less than cost
    if decoded_action - (discount_act + 1) < true_value:
        discount_act = max(decoded_action - true_value, 1) - 1

    # store revealed discount, prefetched action
    self.prefetched_discount = discount_act
    self.prefetched_action = decoded_action

    if truthful_reveal:
        # truthful: info = discount_act
        revealed_info = discount_act
    else:
        # non-truthful: info = info_algo(discount)
        encoded_info = self.info_algorithm.generate_action(obs=discount_act)
        revealed_info = encoded_info % self.info_algorithm.bidding_range

    if not hasattr(self, 'info_history'):
        self.info_history = []
    self.info_history.append(revealed_info)

    # info starts from 1
    return revealed_info + 1


class stgame_greedy_agent(custom_agent):

    def __init__(self, args=None, agent_name='', lowest_value=0, highest_value=10):
        super(stgame_greedy_agent, self).__init__(agent_name, lowest_value, highest_value)

        self.args = args

        self.set_algorithm(
            EpsilonGreedy_auction(
                bandit_n=(self.args.valuation_range * self.args.info_signal_range) * self.args.bidding_range,
                bidding_range=self.args.bidding_range, eps=0.01,
                start_point=int(self.args.exploration_epoch),
                # random.random()),
                overbid=self.args.overbid, step_floor=int(self.args.step_floor),
                signal=self.args.public_signal
                )  # 假设bid 也从分布中取
        )

        self.sp_revenue_record = []
        self.sp_income_record = []

    def receive_observation(self, args, budget, agent_name, agt_idx,
                            extra_info, observation, public_signal,
                            true_value_list, action_history, reward_history, allocation_history):
        obs = dict()
        obs['observation'] = observation['observation']
        if budget is not None:
            budget.generate_budeget(user_id=agent_name)
            obs['budget'] = budget.get_budget(user_id=agent_name)  # observe the new budget

        if args.communication == 1 and agt_idx in args.cm_id and extra_info is not None:
            obs['extra_info'] = extra_info

        if args.public_signal:
            obs['public_signal'] = public_signal

        obs['true_value_list'] = true_value_list
        obs['action_history'] = action_history
        obs['reward_history'] = reward_history
        obs['allocation_history'] = allocation_history

        return obs

    def generate_action(self, obs):
        if hasattr(self, 'prefetched_action'):
            action = self.prefetched_action
        else:
            true_value = self.get_latest_true_value()
            leader_info = obs['leader_info']
            encoded_value = true_value * self.args.info_signal_range + leader_info
            encoded_action = self.algorithm.generate_action(obs=encoded_value)

            # Decode the encoded action
            action = encoded_action % (self.algorithm.bidding_range)

        return action

    def update_policy(self, obs, reward, done=False):
        last_action = self.get_latest_action()
        if done:
            true_value = self.get_latest_true_value()
        else:
            true_value = self.get_last_round_true_value()
        leader_info = obs['leader_info']
        encoded_value = true_value * self.args.info_signal_range + leader_info

        encoded_action = last_action + encoded_value * (self.algorithm.bidding_range)
        self.algorithm.update_policy(encoded_action, reward=reward)

        return

    def test_policy(self, true_value, leader_info=0):
        encoded_value = true_value * self.args.info_signal_range + leader_info
        encoded_action = self.algorithm.generate_action(obs=encoded_value, test=True)

        action = encoded_action % (self.algorithm.bidding_range)
        return action

    def share_info(self):
        return None

    def reset_algo(self):
        # # reset algorithm's decaying parameters
        # assert hasattr(self.algorithm,'t')
        # assert hasattr(self.algorithm,'eps')
        # self.algorithm.t = 0
        # self.algorithm.eps = 0.01        
        self.set_algorithm(EpsilonGreedy_auction(
            bandit_n=(self.args.valuation_range * self.args.info_signal_range) * self.args.bidding_range,
            bidding_range=self.args.bidding_range, eps=0.01,
            start_point=int(self.args.exploration_epoch),
            # random.random()),
            overbid=self.args.overbid, step_floor=int(self.args.step_floor),
            signal=self.args.public_signal
            )  # 假设bid 也从分布中取
                           )

    def clear_record(self):
        self.sp_revenue_record = []
        self.sp_income_record = []
        self.true_value_list = []
        self.action_history = []
        self.reward_history = []
        self.allocation_history = []

    def record_all_type_reward(self, revenue, income):
        self.sp_revenue_record.append(revenue)
        self.sp_income_record.append(income)


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
    #print('bound_method', bound_method)
    return bound_method


def set_leader_algorithm(leader_agt, args):
    # set discount algorithm
    get_discount_algorithm_for_leading_agent(leader_agt, args)
    # assign discount funcs
    bind(leader_agt, generate_discount_for_leading_agent, as_name='generate_discount')
    bind(leader_agt, update_discount_for_leading_agent, as_name='update_discount')
    bind(leader_agt, test_policy_with_discount, as_name='test_with_discount')
    bind(leader_agt, test_discount_policy, as_name='test_discount')
    # set info algorithm
    get_info_algorithm_for_leading_agent(leader_agt, args, init=True)
    # assign info funcs
    bind(leader_agt, share_info_with_discount, as_name='share_info')
    bind(leader_agt, update_info_for_leading_agent, as_name='update_info')
    bind(leader_agt, test_info_policy, as_name='test_info')
    return leader_agt


def init_agents_for_stkbg(trained_agent, leading_agent_name, args):
    from copy import deepcopy
    init_stkbg_agents = {}
    for agent_name, agt in trained_agent.items():
        new_agent = deepcopy(agt)
        init_stkbg_agents[agent_name] = new_agent
    leader_agt = init_stkbg_agents[leading_agent_name]
    set_leader_algorithm(leader_agt, args)
    return init_stkbg_agents


def plot_trained_agents_stkbg(trained_agent_list, leading_agent_name, args, epoch):
    # plot func
    def plot_figure_self_defined(estimation, path, arg_folder, folder_name, figure_name, x_label, y_label, title,
                                 x=None):
        import os
        import matplotlib.pyplot as plt
        save_dir = os.path.join(os.path.join(path, arg_folder), folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for key, results in estimation.items():
            if x is None:
                plt.plot(list(range(len(results))), results, label=key)
            else:
                plt.plot(x, results, label=key, marker="o")
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(save_dir, figure_name))
        plt.show()
        plt.close()
        print(os.path.join(save_dir, figure_name))

    # plot trained results:
    # leader: bid policy with discount & info policy
    # follower: bid policy under different info
    leader_agt = trained_agent_list[leading_agent_name]
    # 1. leader bid policy
    estimation = {'raw_bid': [],'bid_with_discount':[]}
    for true_value in range(args.valuation_range):
        estimation['raw_bid'].append(leader_agt.test_policy(true_value,leader_info=0))
        estimation['bid_with_discount'].append(leader_agt.test_with_discount(true_value))
    plot_figure_self_defined(estimation=estimation, path='./results', arg_folder=args.folder_name,
                             folder_name='{}_stkbg_results'.format(args.exp_id),
                             figure_name='leader_bid_policy_round{}'.format(epoch),
                             x_label='true_value', y_label='bid',
                             title='leader bid policy with discount')
    # 2. leader info policy
    # get all valid discount value
    valid_discounts = set()
    for true_value in range(args.valuation_range):
        valid_discounts.add(leader_agt.test_discount(true_value))
    estimation = {'leader': []}
    valid_discounts = list(valid_discounts)
    for discount in valid_discounts:
        estimation['leader'].append(leader_agt.test_info(discount, truthful=(epoch < 2)))
    valid_infos = list(set(estimation['leader']))
    plot_figure_self_defined(estimation=estimation, path='./results', arg_folder=args.folder_name,
                             folder_name='{}_stkbg_results'.format(args.exp_id),
                             figure_name='leader_info_policy_round{}'.format(epoch),
                             x_label='discount', y_label='info',
                             title='leader info policy', x=valid_discounts)
    # 3. follower bid policy
    for agent_name, agt in trained_agent_list.items():
        if agent_name == leading_agent_name:
            continue
        info_options = valid_infos
        estimation = {}
        for info in info_options:
            results = []
            for true_value in range(args.valuation_range):
                results.append(agt.test_policy(true_value, leader_info=info))
            estimation[agent_name + '_info={}'.format(info)] = results
        plot_figure_self_defined(estimation=estimation, path='./results', arg_folder=args.folder_name,
                                 folder_name='{}_stkbg_results'.format(args.exp_id),
                                 figure_name='{}_bid_policy_round{}'.format(agent_name, epoch),
                                 x_label='true_value', y_label='bid',
                                 title=agent_name + ' bid policy with info')


def get_agents_revenues(trained_agents, args):
    revenues = {}
    for agent_name, agt in trained_agents.items():
        revenues[agent_name] = sum(
            agt.sp_revenue_record[-args.revenue_averaged_stamp:]) / args.revenue_averaged_stamp
    return revenues


def get_agents_incomes(trained_agents, args):
    incomes = {}
    for agent_name, agt in trained_agents.items():
        incomes[agent_name] = sum(
            agt.sp_income_record[-args.revenue_averaged_stamp:]) / args.revenue_averaged_stamp
    return incomes
# def generate_action(self, obs):
#     # print('self', self)
#     # print('obs', obs)
#     true_value = self.get_latest_true_value()
#
#     extra_info = obs['leader_info']
#     if extra_info is not None:
#         encoded_action = self.algorithm.generate_action(obs=true_value,decrease=extra_info)
#     else:
#
#         encoded_action = self.algorithm.generate_action(obs=true_value)
#
#     # Decode the encoded action
#     action = encoded_action % (self.algorithm.bidding_range)
#
#     return action

def plot_all_results_stkbg(platform,revenue,incomes,args,leading_agent_name):
    # platform: list[reward]
    # revenue/income: list[{agent_name:reward}]
    import os
    import matplotlib.pyplot as plt
    save_dir = os.path.join(os.path.join('./results', args.folder_name),'{}_stkbg_results'.format(args.exp_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def plotfunc(results,name,title,length,figure_name):
        for label, data in results.items():
            plt.plot(list(range(length)), data, label=label)
        plt.xlabel('stackelberg epoch')
        plt.ylabel(name)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(save_dir, figure_name))
        plt.show()
        plt.close()
        print(os.path.join(save_dir, figure_name))

    plotfunc({'sell price':platform},'sell price','sell price per epoch',len(platform),'sell_price_per_epoch_stkbg')
    plotfunc({'leader_income':[income_dict[leading_agent_name] for income_dict in incomes]},'income','income per epoch',len(platform),'leader_income_per_epoch_stkbg')
    revenue_plot = {}
    for revenue_dict in revenue:
        for agent_name, r in revenue_dict.items():
            if agent_name != leading_agent_name:
                if agent_name not in revenue_plot:
                    revenue_plot[agent_name] = []
                revenue_plot[agent_name].append(r)
    plotfunc(revenue_plot,'revenue','revenue per epoch',len(platform),'follwer_revenue_per_epoch_stkbg')

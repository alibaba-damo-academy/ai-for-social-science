import matplotlib.pyplot as plt
import os
from agent_utils.signal import *
import seaborn as sns

sns.set_style("darkgrid")


def build_path(save_dir):
    # check path exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return


def build_speicial_estimation(args, agent_list, agt_name, estimation=[]):
    # deal with speical situation
    # eg. in the diversed information agent

    avg_estimation = [0] * (2 * args.public_signal_range + 1)
    cnt = [0] * (2 * args.public_signal_range + 1)

    for signal_1 in range(2 * args.public_signal_range + 1):

        for signal_2 in range(2 * args.public_signal_range + 1):
            state = signal_1 + signal_2 * (2 * args.public_signal_range + 1)

            action = agent_list[agt_name].test_policy(state=state)

            avg_signal = int((signal_1 + signal_2) / 2)

            avg_estimation[avg_signal] += action
            cnt[avg_signal] += 1


    for i in range(2 * args.public_signal_range):
        action = avg_estimation[i] * 1.0 / cnt[i]
        if i == 0:
            estimation[agt_name].append(0.0)
        else:
            estimation[agt_name].append(action * 1.0 / i)



    return estimation


def build_estimation(args, agent_list, env_agent, estimation=[]):
    if args.public_signal == 1:
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
                        if args.test_mini_env==1:
                            estimation[agent].append(action)
                        else:
                            estimation[agent].append(0.0)

                    else:
                        estimation[agent].append(action * 1.0 / partial_signal_value)

                # from signal to partial value
                # function with valuation
                # value_list=[0]*(args.public_signal_range * args.agt_obs_public_signal_dim) # total value with add
                #
                # for state in range(max_state):
                #     value = signal_decode(args,state)
                #     value_list[value] +=


    else:  # no signal
        for i, agent in enumerate(env_agent):
            if (args.assigned_valuation is None) or (i >= len(args.assigned_valuation)):
                highest_value = args.valuation_range
            else:
                highest_value = args.assigned_valuation[i]

            for true_value in range(highest_value):
                action = agent_list[agent].test_policy(true_value)

                if true_value == 0:
                    if args.mechanism == 'second_price' or args.mechanism == 'third_price':
                        estimation[agent].append(1.0)
                    else:
                        estimation[agent].append(0.0)
                else:
                    if args.action_mode == 'div':
                        estimation[agent].append(action * 1.0 / args.div_const)

                    else:
                        estimation[agent].append(action * 1.0 / true_value)

    return estimation

def build_estimation_given_value(args, agent_list, env_agent, estimation=[]):
    # Convert value to state 
    if args.public_signal == 1 and args.public_signal_asym:
        for run in range(200): 
            for i, agent in enumerate(env_agent):
                # max_state = max_signal_value(args)
                state = int(np.random.uniform((1/args.player_num*i)*10, (2-1/args.player_num*i)*10 + 1))
                action = agent_list[agent].test_policy(state=int(state))

                # define the bid / signal
                if args.value_to_signal:
                    partial_signal_value = 1.0
                elif 'kl' in args.folder_name:
                    partial_signal_value = max(1.0, state * 1.0)
                else:
                    partial_signal_value = signal_to_value_sum(signal_decode(args, state))

                if partial_signal_value == 0:
                    estimation[agent].append(0.0)
                else:
                    estimation[agent].append(action * 1.0 / partial_signal_value)
    return estimation

def build_payoff_winprob_estimation(args, agent_list, env_agent, \
    payoff_estimation=[], winprob_estimation = []):
    # Convert value to state 
    
    if args.public_signal == 1 and args.public_signal_asym:
        for value in range(args.valuation_range +1): 
            winprobs = [[] for i in range(len(env_agent))] 
            payoffs = [[] for i in range(len(env_agent))] 
            for run in range(1000): 
                bids = [] 
                for i, agent in enumerate(env_agent):
                    state = int(np.random.uniform((1/args.player_num*i)*value, (2-1/args.player_num*i)*value + 1))
                    action = agent_list[agent].test_policy(state=int(state))
                    bids.append(action)
                order = np.argsort(bids)
                # payoff = [state - bids[order[-2]] if i == order[-1] else 0 for i, _ in enumerate(bids)]
                for i, agent in enumerate(env_agent):
                    payoff = state - bids[order[-2]] if i == order[-1] else 0
                    win = (i == order[-1]) 
                    payoffs[i].append(payoff)
                    winprobs[i].append(win)
            for i, agent in enumerate(env_agent):
                payoff_estimation[agent].append(np.mean(payoffs[i]))
                winprob_estimation[agent].append(np.mean(winprobs[i]))
    return payoff_estimation, winprob_estimation



def get_mechanism_name(args):
    mechanism_list = ['second_price', 'first_price', 'third_price', 'pay_by_submit', 'vcg','deep','customed']

    if args.mechanism in mechanism_list:
        mechanism_name = args.mechanism
    else:
        # self designed mechanism name
        print('self designed mechansim name')
    if args.action_mode == 'div':
        mechanism_name = mechanism_name + '_div_mode_'

    return mechanism_name


def get_figure_name(args, epoch=0, prefix=None):
    if args.overbid:
        overbid_fig_name = 'overbid'
    else:
        overbid_fig_name = 'no_overbid'

    mechanism_name = get_mechanism_name(args)

    epoch_str = 'epoch_' + str(epoch)

    if prefix is None:
        prefix = ''

    figure_name = prefix + 'agt_' + str(args.player_num) + '_' + overbid_fig_name + '_' + mechanism_name + '_' + epoch_str

    return figure_name


def plot_conditional_figure(
        path, folder_name, args, epoch, prefix, agent_list, env_agent, plot_id='cm', mode='max_only'
):
    if plot_id == 'cm':  # plot with communication condition

        if mode == 'max_only':  # only compute the maximal valuation
            estimations = {agent: [] for agent in env_agent}

            for agt_id in args.cm_id:  # plot commununication user policies
                agent_name = env_agent[agt_id]
                agt = agent_list[agent_name]

                save_dir = os.path.join(os.path.join(path, args.folder_name), folder_name)
                build_path(save_dir)

                #
                figure_name = get_figure_name(args, epoch, prefix='info_on_' + str(agent_name))

                # get the maximal observation upperbound
                max_info = agt.history_max  # [1- V+1 ] where 0 represent no info

                plot_info_list = [i * 10 for i in range(int(max_info / 10))]  # plot when observe K infos

                # print(max_info)

                # build estimation

                for plot_info in plot_info_list:
                    labels = agent_name + '|info:' + str(plot_info)

                    estimation = []
                    if (args.assigned_valuation is None) or (agt_id >= len(args.assigned_valuation)):
                        highest_value = args.valuation_range
                    else:
                        highest_value = args.assigned_valuation[agt_id]

                    print('agent' + agent_name + '(' + str(highest_value) + ')  | info ' + str(plot_info))

                    for true_value in range(highest_value):
                        action = agt.test_policy(true_value, assign_information=plot_info)

                        if args.action_mode == 'div':
                            estimation.append(action * 1.0 / args.div_const)

                        else:
                            if true_value == 0:
                                if args.mechanism == 'second_price' or args.mechanism == 'third_price':
                                    estimation.append(1.0)
                                else:
                                    estimation.append(0.0)
                            else:
                                estimation.append(action * 1.0 / true_value)

                    # print(estimation)

                    plt.plot([x for x in range(len(estimation))], estimation, label=str(labels))
                    estimations[agent_name].append(estimation)

                plt.ylim(0, 5.0)
                plt.ylabel('bid/valuation')
                plt.xlabel('valuation')
                plt.legend()
                print(os.path.join(save_dir, figure_name))
                plt.savefig(os.path.join(save_dir, figure_name))
                plt.show()
                plt.close()


def plot_dist_figure(path, folder_name, args, epoch, prefix=None,
                     estimations=[]):  # make extra folder named "first" |"second"

    save_dir = os.path.join(os.path.join(path, args.folder_name), folder_name)
    build_path(save_dir)

    #
    figure_name = get_figure_name(args, epoch, prefix=prefix)
    saved_data = []
    # bid / value
    ax = plt.gca()

    for i in range(args.player_num):
        agt_name = 'player_' + str(i)

        plot_data = estimations[agt_name]
        color = next(ax._get_lines.prop_cycler)['color']
        # plt.hist(plot_data, label=str(agt_name), color = color, bins = 100)
        sns.distplot(plot_data, hist=False, kde=True, label=str(agt_name), bins=100, color=color)
        if not args.public_signal_asym: 
            plt.xlim(0, 2.0)
        plt.axvline(np.mean(plot_data), color=color)
        plt.axvline(np.median(plot_data), ls='--', color=color)

    if args.public_signal_asym: 
        plt.xlabel('bid')
        plt.title('Distribution of Player Bid for v = 10')
    else: 
        plt.xlabel('bid/valuation')
        
    plt.legend()
    print(os.path.join(save_dir, figure_name + '_dist'))
    plt.savefig(os.path.join(save_dir, figure_name + '_dist'))
    plt.show()
    plt.close()


def optimal_bid(signal, nplayer=3, vrange=10):
    x = signal / vrange
    t = nplayer
    result = (-1 / (2 ** t * (t - 2)) + 1 / (2 ** t * (t - 2)) / (0.5 * x) ** (t - 2)) / (
                -1 / (2 ** t * (t - 1)) + 1 / (2 ** t * (t - 1)) / (0.5 * x) ** (t - 1))
    return result * vrange


def plot_figure(path, folder_name, args, epoch, prefix=None,
                estimations=[]):  # make extra folder named "first" |"second"

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


def plot_payoff_winprob_figure(path, folder_name, args, epoch, prefix=None,
                estimations=[]):  # make extra folder named "first" |"second"

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

    if prefix == 'payoff': 
        plt.ylabel('Payoff')
    elif prefix == 'winprob': 
        plt.ylabel('Win Probability')

    plt.xlabel('Public Value')
    plt.legend()
    print(os.path.join(save_dir, figure_name))
    plt.savefig(os.path.join(save_dir, figure_name))
    plt.show()
    plt.close()


def plot_figure2(path, folder_name, args, epoch, prefix=None,
                 estimations=[]):  # make extra folder named "first" |"second"
    save_dir = os.path.join(os.path.join(path, args.folder_name), folder_name)
    build_path(save_dir)

    #
    figure_name = get_figure_name(args, epoch, prefix=prefix)
    saved_data = []
    # bid / value

    for agt_name in estimations.keys():
        plot_data = estimations[agt_name]

        saved_data.append(plot_data)
        plt.plot([x for x in range(len(plot_data))], plot_data, label=str(agt_name))

    plt.ylim(0, 2.0)
    plt.ylabel('bid/valuation')
    plt.xlabel('valuation')
    plt.legend()
    print(os.path.join(save_dir, figure_name))
    plt.savefig(os.path.join(save_dir, figure_name))
    plt.show()
    plt.close()


def print_agent_action(agent, fixed_agt_name, action, true_value, H, args):
    # if args.action_mode=='div':
    #     action = true_value * action *1.0 / args.bidding_range
    if agent == fixed_agt_name:
        # define as bid truthfully

        print('fixed agent decide to bid truthfully: ' + str(action) + '/' + str(H) + ' with his truly value' + str(
            true_value) + '/' + str(H))



    else:
        if (action) == true_value:
            print('smart' + agent + '+ learn to bid truthful: ' + str(action) + '/' + str(
                H) + ' with his truly value' + str(true_value) + '/' + str(H))
        else:
            print(
                'smart' + agent + '+ decide to bid: ' + str(action) + '/' + str(H) + ' with his truly value' + str(
                    true_value) + '/' + str(H))

    print('-----')


def print_agent_reward(agent_list, agent_name_list, avg_epoch=100, filter_allocation=True):
    for agent_name in agent_name_list:
        agent = agent_list[agent_name]
        if filter_allocation:
            print('agent ' + agent_name + ' averaged reward (avg on ' + str(avg_epoch) + ') is ' +
                  str(agent.get_averaged_reward(epoch=avg_epoch))
                  + ' | averaged reward at win is ' + str(agent.get_averaged_reward(epoch=avg_epoch, allocated_only=True))

                  )
        else:
            print('agent ' + agent_name + ' averaged reward (avg on ' + str(avg_epoch) + ') is ' + str(
                agent.get_averaged_reward(epoch=avg_epoch)))

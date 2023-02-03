from plot.plot_util import *
from agent_utils.action_encoding import *


def print_process(env, args, agt_list, new_action, agent_name,
                  next_true_value, epoch, revenue_record,agents_list=None,prefix=None

                  ):
    if agents_list is None:
        agents_list=env.agents

    if (args.print_key):
        if args.action_mode == 'div':
            new_action = action_de_transform(transformed_action=new_action, args=args)
        print_agent_action(agent_name, fixed_agt_name, new_action, true_value=next_true_value,
                           H=args.bidding_range - 1, args=args)
    ##
    
    if ((epoch / args.player_num) % args.estimate_frequent == 0) and (agent_name == agents_list[-1]):
        
        if args.item_num > 1: 
            save_dir = os.path.join(os.path.join('./results', args.folder_name), str(args.exp_id))
            build_path(save_dir)
            revenue_record.plot_avg_revenue(
                args, path='./results', folder_name=str(args.exp_id),
                figure_name=get_figure_name(args, epoch=int(epoch / args.player_num)),
                mechanism_name=get_mechanism_name(args),
            )        
            return 
        
        estimations = {agent: [] for agent in agents_list}
        estimations = build_estimation(args, agent_list=agt_list, env_agent=agents_list, estimation=estimations)

        plot_figure(path='./results', folder_name=str(args.exp_id), args=args, epoch=int(epoch / args.player_num),
                    prefix=prefix,
                    estimations=estimations)
        
        if args.public_signal_asym: 
            estimations_given_value = {agent: [] for agent in agents_list}
            estimations_given_value = build_estimation_given_value(args,\
                agent_list=agt_list, env_agent=agents_list, estimation=estimations_given_value)
            plot_dist_figure(path='./results', folder_name=str(args.exp_id), \
                args=args, epoch=int(epoch / args.player_num),
                prefix=prefix, estimations=estimations_given_value)
            payoff_estimations = {agent: [] for agent in agents_list}
            winprob_estimations = {agent: [] for agent in agents_list}
            payoff_estimations, winprob_estimations = build_payoff_winprob_estimation(args, \
                agent_list=agt_list, env_agent=agents_list, \
                payoff_estimation=payoff_estimations, \
                winprob_estimation=winprob_estimations)
            if prefix is None:
                added_prefix=''
            else:
                add_prefix=prefix

            plot_payoff_winprob_figure(path='./results', folder_name=str(args.exp_id), args=args, epoch=int(epoch / args.player_num),
                    prefix='payoff'+add_prefix,
                    estimations=payoff_estimations)
            plot_payoff_winprob_figure(path='./results', folder_name=str(args.exp_id), args=args, epoch=int(epoch / args.player_num),
                    prefix='winprob'+add_prefix,
                    estimations=winprob_estimations)


        
        # if args.public_signal and args.value_to_signal==0:
        #     plot_dist_figure(path='./results', folder_name=str(args.exp_id), args=args, epoch=int(epoch / args.player_num),
        #                 prefix=None,
        #                 estimations=estimations)

        #
        if args.communication == 1 and len(args.cm_id) > 0:
            plot_conditional_figure(
                path='./results', folder_name=str(args.exp_id), args=args, epoch=int(epoch / args.player_num),
                prefix=prefix,
                agent_list=agt_list, env_agent=agents_list
            )

        #
        revenue_record.plot_avg_revenue(
            args, path='./results', folder_name=str(args.exp_id),
            figure_name=get_figure_name(args, epoch=int(epoch / args.player_num)),
            mechanism_name=get_mechanism_name(args),
            # plot_y_range=[0.4*args.valuation_range,args.valuation_range] #plot range
        )

        if args.record_efficiency == 1:
            revenue_record.plot_avg_efficiency(
                args, path='./results', folder_name=str(args.exp_id),
                figure_name=get_figure_name(args, epoch=int(epoch / args.player_num)),
                mechanism_name=get_mechanism_name(args),
                # plot_y_range=[0.4*args.valuation_range,args.valuation_range] #plot range
            )

        # print revenue
        print_agent_reward(agent_list=agt_list, agent_name_list=agents_list, avg_epoch=args.revenue_averaged_stamp,
                           filter_allocation=True
                           )

        print('the latest avg revenue is ' + str(revenue_record.get_last_avg_revenue()))

def record_step(args, agt, allocation, agent_name, epoch, env, reward, revenue_record,agents_list=None):
    if agents_list is None:
        agents_list = env.agents

    # if record allocation:
    agt.record_allocation(allocation=allocation)
    # if record efficiency:
    if args.record_efficiency == 1:
        revenue_record.record_efficiency(allocation=allocation, true_value=agt.get_last_round_true_value(),
                                         epoch=int(epoch / args.player_num), agent_name=agent_name,
                                         end_flag=(agent_name == agents_list[-1]))
    
    # Mode for computing the revenue
    mode = args.mechanism 
    if args.item_num > 1: 
        mode = "multi_item"
    revenue_record.record_revenue(allocation=allocation, pay=reward, epoch=int(epoch / args.player_num),
                                  mode=mode, end_flag=(agent_name == agents_list[-1])
                                  )
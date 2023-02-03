def communication(env, args, agt_list, agent_name):
    if args.communication == 0:
        return None

    extra_info = {agent: {} for agent in env.agents}
    if args.communication_type == 'value':
        if len(args.cm_id) < 1:
            return None
        else:
            for id in args.cm_id:
                i = env.agents[id]
                extra_info[i]['value'] = agt_list[i].get_latest_true_value()
                extra_info[i]['others_value'] = []
                for other_id in args.cm_id:
                    if other_id != id:
                        # others :
                        j = env.agents[other_id]
                        if agt_list[j].get_latest_true_value() is not None:
                            extra_info[i]['others_value'].append(agt_list[j].get_latest_true_value())

            return extra_info

def submit_info_to_env(args, agt_idx, test_policy, agent_name, next_true_value):
    if args.inner_cooperate == 1 and agt_idx in args.inner_cooperate_id:
        # print( agent_name+'communicate infos with '+ str(next_true_value))
        test_policy.set_information(agent=agent_name, infos=next_true_value)  # pass the next true value

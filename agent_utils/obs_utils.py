def receive_observation(args, budget, agent_name, agt_idx, extra_info, observation,public_signal):
    obs = observation['observation']
    if budget is not None:
        # generate  new budget
        budget.generate_budeget(user_id=agent_name)
        obs = budget.get_budget(user_id=agent_name)  # observe the new budget

    if args.communication == 1 and agt_idx in args.cm_id and extra_info is not None:
        obs = extra_info[agent_name]['others_value']

    if args.public_signal :
        obs = public_signal

    return obs

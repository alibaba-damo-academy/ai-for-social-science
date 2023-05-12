def reinit_env_for_stkbg(dynamic_env, agents):
    dynamic_env.re_init()
    dynamic_env.run_full_stackelberg()
    for agent_name, agt in agents.items():
        dynamic_env.set_rl_agent(rl_agent=agt, agent_name=agent_name)
    dynamic_env.enable_discount()
    dynamic_env.enable_information()
    # donot update leader bid policy (update discount or info)
    dynamic_env.turnoff_leading_agent_update()
    dynamic_env.adjust_reward_mode(reward_mode='income')


def set_env_update_mode(dynamic_env, epoch):
    if epoch % 2 == 0:
        print('leader update...')
        # leader update
        if epoch % 4 == 0:
            # update discount
            print('update discount policy')
            dynamic_env.turnon_leader_discount_update()
            dynamic_env.turnoff_leader_info_update()
            # turn off follower update
            dynamic_env.turnoff_follower_agent_update()
        else:
            # update info
            print('update information policy')
            dynamic_env.turnoff_leader_discount_update()
            dynamic_env.turnon_leader_info_update()
            # turn off follower update
            dynamic_env.turnoff_follower_agent_update()
    else:
        print('follower update...')
        # follwer update
        # turn off leader update
        print('update bid policy')
        dynamic_env.turnoff_leader_discount_update()
        dynamic_env.turnoff_leader_info_update()
        # turn on follower update
        dynamic_env.turnon_follower_agent_update()

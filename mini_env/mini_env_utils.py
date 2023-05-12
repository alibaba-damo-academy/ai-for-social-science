import random
import copy



def increase_info_to_obs(obs,extra_info_name='true_value',value=0):

    if obs is None:
        obs = dict()

    obs[extra_info_name]=value

    return obs

def set_agent_signal_observation(
        public_signal_generator=None,
        agent_name='',
        agt=None,
        mode='default'
):
    if public_signal_generator is None:
        return None

    #use default partial observation list
    if mode =='default' :
        return public_signal_generator.get_partial_signal_realization(
            observed_dim_list = agt.get_public_signal_dim_list()
        )
    elif mode =='all':
        #denote as return all signal
        return public_signal_generator.get_whole_signal_realization()

    elif mode =='modify':
        #apply noise or other method
        original_signal = public_signal_generator.get_partial_signal_realization(
            observed_dim_list = agt.get_public_signal_dim_list()
        )
        if agent_name =='??':
            noise = 1
        else:
            noise=0


        return original_signal + noise
    else:
        return None

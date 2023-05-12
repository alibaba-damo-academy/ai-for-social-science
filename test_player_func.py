def get_custom_agent_name():
    return 'player_3'
            
def generate_action(self, obs):
    # print('self', self)
    # print('obs', obs)
    encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

    # Decode the encoded action 
    action = encoded_action % (self.algorithm.bidding_range)

    return action

def update_policy(self, obs, reward, done):
    last_action = self.get_latest_action()
    if done:
        true_value = self.get_latest_true_value()
    else:
        true_value = self.get_last_round_true_value()
    encoded_action = last_action + true_value * (self.algorithm.bidding_range)
    self.algorithm.update_policy(encoded_action, reward=reward)

    return

def receive_observation(self, args, budget, agent_name, agt_idx, 
                        extra_info, observation, public_signal,
                        true_value_list, action_history, reward_history, allocation_history):
    obs = dict()
    obs['observation'] = observation['observation']
    if budget is not None:
        budget.generate_budeget(user_id=agent_name)
        obs['budget'] = budget.get_budget(user_id=agent_name)  # observe the new budget

    if args.communication == 1 and agt_idx in args.cm_id and extra_info is not None:
        obs['extra_info'] = extra_info[agent_name]['others_value']

    if args.public_signal:
        obs['public_signal'] = public_signal

    return obs
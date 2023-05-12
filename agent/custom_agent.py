import random
from rl_utils.solver import *

class custom_agent(object):

    def __init__(self, agent_name='', lowest_value=0, highest_value=10):
        self.agent_name = agent_name
        self.true_value_list=[]  # True value of item for the agent 
        self.action_history = [] 
        self.reward_history = [] 
        self.allocation_history=[] # list of agent id of allocated items
        self.payment_history = []


        self.lowest_value = lowest_value
        self.highest_value = highest_value
        self.last_obs=None
        self.record_data_num=0

        self.algorithm = None # Agent by the agent 
        self.policy = None  # Policy refers to the allocation policy

    ##
    def assign_generate_true_value(self, generate_true_value):
        self.generate_true_value = generate_true_value
        return

    ## get ###

    def get_last_action(self):
        if len(self.action_history) > 1:
            return self.action_history[-2]
        else:

            return -1

    def get_latest_action(self):
        if len(self.action_history) > 0:
            return self.action_history[-1]
        else:

            return -1
    def get_latest_true_value(self):
        return self.true_value_list[-1]

    def get_last_round_true_value(self):
        return self.true_value_list[-2] # update to -2 as the -1 is the next round true value self.true_value_list[-1

    def get_action_history(self):
        return self.action_history

    def get_reward_history(self):
        return self.reward_history

    def get_true_value_history(self):
        return self.true_value_list
    def get_allocation_history(self):
        return self.allocation_history

    def get_allocation_epoch_history(self,epoch): 
        # history of the allocation epochs, used for plotting
        return self.allocation_history[-epoch:]


    def get_averaged_reward(self,epoch=100,allocated_only=False):
        """
        Get the averaged reward for the previous (allocated) epochs
        """
        if epoch ==0 :
            return 0
        reward =sum(self.reward_history[-epoch:])
        if allocated_only:
            allocation_his = sum(self.get_allocation_epoch_history(epoch))
            if allocation_his ==0:
                return 0
            else:

                return reward * 1.0 / allocation_his
        else:
            return reward * 1.0 / min(epoch,len(self.reward_history))


    def get_last_obs(self):
        return self.last_obs

    ###  set ###
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
    
    def set_generate_action(self, generate_action):
        self.generate_action = generate_action
        # import inspect
        # lines = inspect.getsource(self.generate_action)
        # print(lines)

    def set_update_policy(self, update_policy):
        self.update_policy = update_policy

    def set_receive_observation(self, receive_observation):
        self.receive_observation = receive_observation

    def set_policy(self, policy):
        self.policy = policy
        return

    ##  record ###
    def record_reward(self,reward):
        self.reward_history.append(reward)
        return

    def record_true_value(self,value):

        self.true_value_list.append(value)
        return
    def record_action(self,action):

        self.action_history.append(action)
        return
    def record_allocation(self,allocation):
        self.allocation_history.append(allocation)

    def record_obs(self,obs):
        self.last_obs=obs

    def test_policy(self, true_value):
        return true_value

    def record_payment(self,payment):
        self.payment_history.append(payment)



class simple_greedy_agent(custom_agent):

    def __init__(self, args=None, agent_name='', lowest_value=0, highest_value=10):
        super(simple_greedy_agent, self).__init__(agent_name, lowest_value, highest_value)

        self.args = args

        self.set_algorithm(
            EpsilonGreedy_auction(bandit_n=(highest_value + 1) * self.args.bidding_range,
                                  bidding_range=self.args.bidding_range, eps=0.01,
                                  start_point=int(self.args.exploration_epoch),
                                  # random.random()),
                                  overbid=self.args.overbid, step_floor=int(self.args.step_floor),
                                  signal=self.args.public_signal
                                  )  # 假设bid 也从分布中取
        )

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

        obs['true_value_list'] = true_value_list
        obs['action_history'] = action_history
        obs['reward_history'] = reward_history
        obs['allocation_history'] = allocation_history
        
        return obs

    def generate_action(self, obs):
        encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

        # Decode the encoded action 
        action = encoded_action % (self.algorithm.bidding_range)

        return action
    
    def update_policy(self, obs, reward, done=False):
        last_action = self.get_latest_action()
        if done:
            true_value = self.get_latest_true_value()
        else:
            true_value =self.get_last_round_true_value()

        encoded_action = last_action + true_value * (self.algorithm.bidding_range)
        self.algorithm.update_policy(encoded_action,reward=reward)

        return

    def test_policy(self, true_value):
        encoded_action = self.algorithm.generate_action(obs=true_value,test=True)

        action = encoded_action % (self.algorithm.bidding_range)
        return action        

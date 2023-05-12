from __future__ import division

import numpy as np
import time
from scipy.stats import beta
import random
import copy

class base_agent(object):

    def __init__(self, agent_name=''):
        self.agent_name = agent_name
        self.true_value_list=[]  # True value of item for the agent 
        self.action_history = [] 
        self.reward_history = [] 
        self.allocation_history=[] # list of agent id of allocated items 

        self.algorithm = None # Agent by the agent 
        self.policy = None  # Policy refers to the allocation policy
    ##
    def generate_action(self,observation):
        return
    def generate_true_value(self):
        return
    def update_policy(self,state,reward):
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
    ###  set ###
    def set_algorithm(self, algorithm, item_num = 1):
        if item_num == 1: 
            self.algorithm = algorithm
        else: 
            self.algorithm = [copy.deepcopy(algorithm) for _ in range(item_num)]
        return

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

class fixed_agent(base_agent):
    """
    Fixed learner places its true value as its bid
    """

    def __init__(self, agent_name='', lowest_value=0,highest_value=10, item_num = 1):
        super(fixed_agent, self).__init__(agent_name)
        # highest posible true value of the item for an agent
        self.highest_value=highest_value
        # lowest posible true value of the item for an agent
        self.lowest_value=lowest_value
        self.item_num = item_num 

    def generate_true_value(self,method='uniform'):
        """
        Generate the true value of the auction item for an agent
        """ 
        if self.item_num == 1: 
            if method =='uniform':
                true_value= random.randint(self.lowest_value,self.highest_value)  #[0-H]
            elif method =='gaussian': # Wrong implementation
                true_value = int(np.random.uniform(self.lowest_value, self.highest_value+1))
            #else:
            #    true_value
                # print(true_value)# remained for other generation method
        else: 
            if method =='uniform':
                true_value= [random.randint(self.lowest_value,self.highest_value) for i in range(self.item_num) ]  #[0-H]
            elif method =='gaussian': # Wrong implementation
                true_value = [int(np.random.uniform(self.lowest_value, self.highest_value+1)) for i in range(self.item_num)] 

        self.record_true_value(true_value)
        return

    def generate_action(self,observation):
        """
        Fixed learner always bid its true value
        """
        action = self.get_latest_true_value()
        self.record_action(action)


        return action


    def update_policy(self,state,reward, done = False):
        self.record_reward(reward)

        return

    def test_policy(self,true_value):

        return true_value

class greedy_agent(base_agent):
    """
    Greedy agent generates its action based on certain greedy 'self.algorithm'. 
    """ 
    def __init__(self, agent_name='', lowest_value=0, highest_value=10, item_num = 1):
        super(greedy_agent, self).__init__(agent_name)
        self.highest_value = highest_value
        self.lowest_value = lowest_value
        self.item_num = item_num

    def generate_true_value(self, method='uniform', true_value = None):

        # Generate the true value given the true value
        if true_value is not None: 
                self.record_true_value(true_value)
                return
        elif self.item_num == 1: 
            if method == 'uniform':
                true_value = random.randint(self.lowest_value, self.highest_value)  # [0-H]
            elif method == 'gaussian':
                true_value = int(np.random.uniform(self.lowest_value, self.highest_value + 1))
            else:
                true_value = -1  # remained for other generation method
        else: 
            if method =='uniform':
                true_value= [random.randint(self.lowest_value,self.highest_value) for i in range(self.item_num) ]  #[0-H]
            elif method =='gaussian': # Wrong implementation
                true_value = [int(np.random.uniform(self.lowest_value, self.highest_value+1)) for i in range(self.item_num)] 

        self.record_true_value(true_value)
        return

    def generate_action(self,observation):
        
        if self.item_num  == 1: 
            # Generate the action given its observed true value of the item
            encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

            # Decode the encoded action 
            action = encoded_action % (self.algorithm.bidding_range)
        else: 
            action = []
            for i, true_value in enumerate(self.get_latest_true_value()): 
                # Generate the action given its observed true value of the item
                encoded_action = self.algorithm[i].generate_action(obs=true_value)
                # Decode the encoded action 
                action.append(encoded_action % (self.algorithm[i].bidding_range))

        self.record_action(action)  # to guoyang: you left out the record_action ->multi-item =list | single item =int

        return action

    def update_policy(self, state, reward,done=False):

        last_action = self.get_latest_action()
        if done:
            true_value = self.get_latest_true_value()
        else:
            true_value =self.get_last_round_true_value()
        if self.item_num == 1: 
            encoded_action = last_action + true_value * (self.algorithm.bidding_range)
            self.algorithm.update_policy(encoded_action,reward=reward)
        else: 
            for i, true_val in enumerate(true_value): 
                #ori ver : encoded_action = last_action + true_val * (self.algorithm[i].bidding_range)
                #eric added: last action should be record
                encoded_action = last_action[i] + true_val * (self.algorithm[i].bidding_range)

                self.algorithm[i].update_policy(encoded_action,reward=reward)

        self.record_reward(reward)

        return


    def encode(self,last_state):
        """
        Encode the last state (a value,action tuple) as a single number. 

        Encode the output the RL agent so that the RL algorithm can run on different interface. 
        For example, change the price, or price higher than other agent.
        """

        [last_true_value, last_action] = last_state  # [1-H, 0-H-1]
        res = (last_true_value) * (self.algorithm.bidding_range) + last_action  # 观测1-10块钱，但是可以出价0.1- 10.0 所以100个切分

        return res

    def test_policy(self, true_value):
        encoded_action = self.algorithm.generate_action(obs=true_value,test=True)

        action = encoded_action % (self.algorithm.bidding_range)
        return action




class dived_greedy_agent(base_agent):

    def __init__(self, agent_name='', lowest_value=0, highest_value=10):
        super(dived_greedy_agent, self).__init__(agent_name)
        self.highest_value = highest_value
        self.lowest_value = lowest_value

    def generate_true_value(self, method='uniform'):

        if method == 'uniform':
            true_value = random.randint(self.lowest_value, self.highest_value)  # [0-H]
        elif method == 'gaussian':
            true_value = int(np.random.uniform(self.lowest_value, self.highest_value + 1))
        else:
            true_value = -1  # remained for other generation method

        self.record_true_value(true_value)
        return

    def generate_action(self,observation):

        encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

        action = encoded_action % (self.algorithm.bidding_range)


        self.record_action(action)

        # Why compute this value

        return self.get_latest_true_value() * action*1.0 / self.algorithm.bidding_range
        # V * (x / bidding_range) x=[0~bidding_range]

    def update_policy(self, state,reward,done=False):

        last_action = self.get_latest_action()
        if done:
            true_value = self.get_latest_true_value()
        else:
            true_value = self.get_last_round_true_value()

        encoded_action = last_action + true_value * (self.algorithm.bidding_range)

        self.algorithm.update_policy(encoded_action,reward=reward)

        self.record_reward(reward)

        return

    def encode(self,last_state):
        [last_true_value, last_action] = last_state  # [1-H, 0-H-1]
        res = (last_true_value) * (self.algorithm.bidding_range) + last_action  # 观测1-10块钱，但是可以出价0.1- 10.0 所以100个切分

        return res

    def test_policy(self, true_value):
        '''
        Test the action generation of the greedy learner without recording the action.
        '''
        encoded_action = self.algorithm.generate_action(obs=true_value,test=True)

        action = encoded_action % (self.algorithm.bidding_range)

        return action


class greedy_common_value_agent(base_agent):
    """
    Greedy agent generates its action based on certain greedy 'self.algorithm'. 
    """ 
    def __init__(self, agent_name='', lowest_value=0, highest_value=10):
        super(greedy_common_value_agent, self).__init__(agent_name)
        self.highest_value = highest_value
        self.lowest_value = lowest_value

    def generate_true_value(self, method='uniform', true_value = -1):

        # Generate the true value given the true value
        if method == 'uniform':
            true_value = random.randint(self.lowest_value, self.highest_value)  # [0-H]
        elif method == 'gaussian':
            true_value = int(np.random.uniform(self.lowest_value, self.highest_value + 1))
        #elif method == "fixed": 
        #    print(true_value)

        # Otherwise, use the fixed value for true value

        self.record_true_value(true_value)
        return

    def generate_action(self,observation):        
        # Generate the action given its observed true value of the item
        encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

        # Decode the encoded action 
        action = encoded_action % (self.algorithm.bidding_range)


        self.record_action(action)

        return action

    def update_policy(self, state,reward,done=False):

        last_action = self.get_latest_action()
        encoded_action = last_action + self.get_last_round_true_value() * (self.algorithm.bidding_range)

        self.algorithm.update_policy(encoded_action,reward=reward)

        self.record_reward(reward)

        return

    def encode(self,last_state):
        """
        Encode the last state (a value,action tuple) as a single number. 

        Encode the output the RL agent so that the RL algorithm can run on different interface. 
        For example, change the price, or price higher than other agent.
        """

        [last_true_value, last_action] = last_state  # [1-H, 0-H-1]
        res = (last_true_value) * (self.algorithm.bidding_range) + last_action  # 观测1-10块钱，但是可以出价0.1- 10.0 所以100个切分

        return res

    def test_policy(self, true_value):
        encoded_action = self.algorithm.generate_action(obs=true_value,test=True)

        action = encoded_action % (self.algorithm.bidding_range)


        return action


class greedy_agent_know_upbound(base_agent):
    """
    Greedy agent generates its action based on certain greedy 'self.algorithm'.
    This agent type knows his upbound before bidding

    """

    def __init__(self, agent_name='', lowest_value=0, highest_value=10):
        super(greedy_agent_know_upbound, self).__init__(agent_name)
        self.highest_value = highest_value
        self.lowest_value = lowest_value

    def generate_true_value(self, method='uniform', true_value=None):

        # Generate the true value given the true value
        if true_value is not None:
            self.record_true_value(true_value)
            return
        elif method == 'uniform':
            true_value = random.randint(self.lowest_value, self.highest_value)  # [0-H]
        elif method == 'gaussian':
            true_value = int(np.random.uniform(self.lowest_value, self.highest_value + 1))
        else:
            true_value = -1  # remained for other generation method

        self.record_true_value(true_value)
        return

    def generate_action(self, observation):

        # Generate the action given its observed true value of the item

        # upper_bound should be int and not exceed the bidding range
        encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value(),upper_bound=int(observation) )

        # Decode the encoded action
        action = encoded_action % (self.algorithm.bidding_range)

        self.record_action(action)

        return action

    def update_policy(self, state, reward):

        last_action = self.get_latest_action()
        encoded_action = last_action + self.get_last_round_true_value() * (self.algorithm.bidding_range)

        self.algorithm.update_policy(encoded_action, reward=reward)

        self.record_reward(reward)

        return

    def encode(self, last_state):
        """
        Encode the last state (a value,action tuple) as a single number.

        Encode the output the RL agent so that the RL algorithm can run on different interface.
        For example, change the price, or price higher than other agent.
        """

        [last_true_value, last_action] = last_state  # [1-H, 0-H-1]
        res = (last_true_value) * (self.algorithm.bidding_range) + last_action  # 观测1-10块钱，但是可以出价0.1- 10.0 所以100个切分

        return res

    def test_policy(self, true_value):
        encoded_action = self.algorithm.generate_action(obs=true_value, test=True)

        action = encoded_action % (self.algorithm.bidding_range)

        return action


class greedy_agent_budget_sample_version(base_agent):
    """
    Greedy agent generates its action based on certain greedy 'self.algorithm'.
    This agent type knows his upbound before bidding

    """

    def __init__(self, agent_name='', lowest_value=0, highest_value=10,budget_range=10):
        super(greedy_agent_budget_sample_version, self).__init__(agent_name)
        self.highest_value = highest_value
        self.lowest_value = lowest_value

        self.budget_range=int(budget_range)
        self.algorithm_list=[None] * self.budget_range

    def set_algorithm(self,algorithm):
        #deep copy mulitply algorithm
        for i in range(self.budget_range):
            self.algorithm_list[i]= copy.deepcopy(algorithm)


        return

    def generate_true_value(self, method='uniform', true_value=None):

        # Generate the true value given the true value
        if true_value is not None:
            self.record_true_value(true_value)
            return
        elif method == 'uniform':
            true_value = random.randint(self.lowest_value, self.highest_value)  # [0-H]
        elif method == 'gaussian':
            true_value = int(np.random.uniform(self.lowest_value, self.highest_value + 1))
        else:
            true_value = -1  # remained for other generation method

        self.record_true_value(true_value)
        return

    def generate_action(self, observation):

        # Generate the action given its observed true value of the item

        #current observation is the budget

        cur_budget = int(observation)

        #assign self.algorithm for the later updating
        self.algorithm = self.algorithm_list[cur_budget]

        # upper_bound should be int and not exceed the bidding range
        encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

        # Decode the encoded action
        action = encoded_action % (self.algorithm.bidding_range)

        self.record_action(action)

        return action

    def update_policy(self, state, reward):

        last_action = self.get_latest_action()
        encoded_action = last_action + self.get_last_round_true_value() * (self.algorithm.bidding_range)

        self.algorithm.update_policy(encoded_action, reward=reward)

        self.record_reward(reward)

        return

    def encode(self, last_state):
        """
        Encode the last state (a value,action tuple) as a single number.

        Encode the output the RL agent so that the RL algorithm can run on different interface.
        For example, change the price, or price higher than other agent.
        """

        [last_true_value, last_action] = last_state  # [1-H, 0-H-1]
        res = (last_true_value) * (self.algorithm.bidding_range) + last_action  # 观测1-10块钱，但是可以出价0.1- 10.0 所以100个切分

        return res

    def test_policy(self, true_value):


        encoded_action = self.algorithm_list[true_value].generate_action(obs=true_value, test=True)

        action = encoded_action % (self.algorithm_list[true_value].bidding_range)

        return action




class greedy_agent_cm_version(base_agent):
    """
    Greedy agent generates its action based on certain greedy 'self.algorithm'.
    This agent type knows his upbound before bidding

    """

    def __init__(self, agent_name='', lowest_value=0, highest_value=10,bidding_range=10,cm_number=2,mode='max_only'):
        super(greedy_agent_cm_version, self).__init__(agent_name)
        self.highest_value = highest_value
        self.lowest_value = lowest_value

        self.communication_agent_number=cm_number
        self.bidding_range = bidding_range
        self.history_max=0

        self.suppose_mode=mode
        if self.suppose_mode == 'multiple':
            self.suppose_number=self.bidding_range ** self.communication_agent_number +1 # v1=10 in [0,30],v2=20 in [0,40] ,state=[10,20] --》 K^id
        elif self.suppose_mode=='max_only':
            self.suppose_number = self.bidding_range+1 #set 0 for non information
        else:
            self.suppose_number = self.communication_agent_number+1

        self.algorithm_list=[None] * self.suppose_number

    def set_algorithm(self,algorithm):
        #deep copy mulitply algorithm
        for i in range(self.suppose_number):
            self.algorithm_list[i]= copy.deepcopy(algorithm)


        return

    def generate_true_value(self, method='uniform', true_value=None):

        # Generate the true value given the true value
        if true_value is not None:
            self.record_true_value(true_value)
            return
        elif method == 'uniform':
            true_value = random.randint(self.lowest_value, self.highest_value)  # [0-H]
        elif method == 'gaussian':
            true_value = int(np.random.uniform(self.lowest_value, self.highest_value + 1))
        else:
            true_value = -1  # remained for other generation method

        self.record_true_value(true_value)
        return

    def generate_action(self, observation):

        # Generate the action given its observed true value of the item and others true value

        # now deal with the simplest one: only consider the max value
        if self.suppose_mode=='max_only':
            others_value = max(observation) +1
            self.history_max=max(self.history_max,others_value)
        else:
            others_value =0

        #assign self.algorithm for the later updating
        self.algorithm = self.algorithm_list[others_value]

        # upper_bound should be int and not exceed the bidding range
        encoded_action = self.algorithm.generate_action(obs=self.get_latest_true_value())

        self.algorithm_list[0].record_action(encoded_action)


        # Decode the encoded action
        action = encoded_action % (self.algorithm.bidding_range)

        self.record_action(action)

        return action

    def update_policy(self, state, reward):

        last_action = self.get_latest_action()
        encoded_action = last_action + self.get_last_round_true_value() * (self.algorithm.bidding_range)

        #print('now updating '+str(last_action)+ ' with reward' + str(reward) +' as his true value is '+str(self.get_last_round_true_value()))

        self.algorithm.update_policy(encoded_action, reward=reward)

        self.algorithm_list[0].update_policy(encoded_action, reward=reward) #suppose not wintness


        self.record_reward(reward)

        return

    def encode(self, last_state):
        """
        Encode the last state (a value,action tuple) as a single number.

        Encode the output the RL agent so that the RL algorithm can run on different interface.
        For example, change the price, or price higher than other agent.
        """

        [last_true_value, last_action] = last_state  # [1-H, 0-H-1]
        res = (last_true_value) * (self.algorithm.bidding_range) + last_action  # 观测1-10块钱，但是可以出价0.1- 10.0 所以100个切分

        return res

    def test_policy(self, true_value,assign_information=None):

        #
        if assign_information is not None:

            info = assign_information +1

        if assign_information is not None and info <=self.history_max :

            encoded_action = self.algorithm_list[info].generate_action(obs=true_value, test=True)
        else:
            if 0:
                encoded_action = self.algorithm_list[self.history_max].generate_action(obs=true_value, test=True) #超过历史记录=没出现过的information，按最高记，或无information尝试
            else:
                encoded_action = self.algorithm_list[0].generate_action(obs=true_value, test=True) #认为无information

        action = encoded_action % (self.algorithm_list[true_value].bidding_range)

        return action
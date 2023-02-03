from __future__ import division

import numpy as np
import time
from scipy.stats import beta
import random

class Solver(object):
    def __init__(self, bandit_n):
        """
        bandit (Bandit): the target bandit to solve.
        """

        self.bandit_n = bandit_n

        self.counts = [0] * self.bandit_n
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.
        self.reward = 0. # Cumulative reward
        self.reward_list = []
        self.last_obs_list=[]
        self.state_list=[]
        self.cost_list=[]
        self.step_floor=0

        self.gamma=1
        self.w=1

        self.uid=-1

    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_proba - self.bandit.probas[i]
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        """
        Run multiple steps of the learning algorithm. Not being used currently.
        """
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)

    # eric
    ## record result
    def record_last_obs(self,last_obs):
        self.last_obs_list.append(last_obs)
        return

    def record_cost(self,cost):
        self.cost_list.append(cost)
        return

    def record_reward(self, reward):

        self.reward = reward

        self.reward_list.append(reward)
    def record_action(self, i):
        self.counts[i] += 1
    def record_state(self,state):
        self.state_list.append(state)
        return
    def set_step_floor(self,step_floor):
        self.step_floor=step_floor

    ##
    def update_reward(self, reward):
        self.reward = reward
        return

    def generate_action(self):
        raise NotImplementedError

    def update_policy(self):
        raise NotImplementedError

    def get_w(self):
        return self.w

    def get_gamma(self):
        return self.gamma

    def get_uid(self):
        return self.uid
    def get_latest_action(self):
        if len(self.actions) == 0:
            return -1
        else:
            return self.actions[-1]
    def get_last_obs(self):
        if len(self.last_obs_list) == 0:
            return 3
        else:
            return self.last_obs_list[-1]


    def get_latest_reward(self):
        if len(self.reward_list) == 0:
            return -1
        else:
            return self.reward_list[-1]

    def get_last_state(self):
        if len(self.state_list) == 0:
            return [-1,-1]
        else:
            return self.state_list[-1]

class EpsilonGreedy_auction(Solver):
    def __init__(self,bandit_n,bidding_range=100, eps=0.1, start_point=0,overbid=False,init_proba=0.0,step_floor=10000,signal=False):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """

        # Bandit_n refers to the single players

        super(EpsilonGreedy_auction, self).__init__(bandit_n)

        assert 0. <= eps <= 1.0
        self.eps = eps
        self.t=0
        self.start_point=start_point # Number of exploration steps

        self.bidding_range=bidding_range
        self.overbid=overbid
        self.signal=signal

        self.true_value_list=[]

        self.decay = 0.5 #3 * self.bandit_n #ori: 10
        self.step_floor=step_floor # Number of steps to eval the avg. maybe eval_step_size
        self.estimates = [init_proba] * self.bandit_n  # Optimistic initialization
        # list len = n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.eps:
            # Let's do random exploration!
            i = np.random.randint(0, self.bandit_n)
        else:
            # Pick the best one.
            i = max(range(self.bandit_n), key=lambda x: self.estimates[x])

        # get reward
        r = self.bandit.generate_reward(i)
        #

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i

    # eric

    def generate_action(self,obs=None,test=False,upper_bound=None):
        # Obs refers to the true value

        # Lower bid does not refer to the actual lower bid, but the lower bid on the 
        # encoded space.
        if upper_bound is not None:
            lower_bid = obs * self.bidding_range
            upper_bid = obs*self.bidding_range + upper_bound+1

            #print(lower_bid)
            #print(upper_bid)


        elif self.overbid or self.signal:
            lower_bid = obs * self.bidding_range
            upper_bid = (obs+1)*self.bidding_range
        else:
            lower_bid=obs*self.bidding_range
            upper_bid =obs*self.bidding_range+obs+1     #  over bid: (obs+1)*self.bidding_range

        if (np.random.random() < self.eps or self.t<self.start_point ) and (test==False):
            # Let's do random exploration!
            #i = np.random.randint(0, self.bandit_n)
            # updated to the m

            if obs is None:
                action_possibility = np.array([ 1/(num+1e-6) for num in self.counts])
                action_possibility /=  sum(action_possibility)
                i = np.random.choice([i for i in range(self.bandit_n)], p=action_possibility.ravel())
            else:
                # allow over bid
                i = np.random.choice([i for i in range(lower_bid,upper_bid)])
                # forbid over bid :


            #print('rnd!!!!!!!!')
        elif obs is not None:
            # Pick the best one.
            #i = max(range(self.bandit_n), key=lambda x: self.estimates[x])

            # with shuffle
            select_estimate=self.estimates[lower_bid : upper_bid]
            max_value = max(select_estimate)

            max_act_list = [k for k, v in enumerate(select_estimate) if v == max_value]
            random.shuffle(max_act_list)

            i=max_act_list[0] +lower_bid #fixed bug in 11.3
            # print(i-lower_bid)
            # print(i)
            # print('---')
        else:
            max_act_list = [k for k, v in enumerate(self.estimates) if v == max(self.estimates)]
            random.shuffle(max_act_list)
            i = max_act_list[0]
        # record
        if not test:

            self.record_action(i)

        return i

    def update_policy(self, i,reward):  # do  record_action(self,i) before update policy

        # self.estimates[i] += 1. / (self.counts[i] + 1) * (self.reward - self.estimates[i])
        # old version as cnt+1 after update policy
        #print('at round' + str(self.t) +'  optimize is ' + str( (self.reward - self.estimates[i])) )
        #if (self.reward - self.estimates[i]) !=0:
            #print('weighted is :' + str(1. / (self.counts[i] + 1)))

        #self.estimates[i] += 1. / (self.counts[i] + 1) * (self.reward - self.estimates[i])

        self.estimates[i] = (reward +self.estimates[i] * self.counts[i]) / (self.counts[i]+1)

        #self.estimates[i] = (self.reward + self.estimates[i]) /2

        #self.estimates[i] = (self.reward *0.1+  0.9 * self.estimates[i])


        self.t+=1




        if self.t ==self.start_point :
            # remove the accumalative counts for optimization
            self.counts=[1] * self.bandit_n


        self.eps= 0.5 **(self.t /  self.step_floor)

        

class Multi_item_EpsilonGreedy_(Solver):
    def __init__(self,bandit_n, item_num = 5, bidding_range=100, eps=0.1, start_point=0,overbid=False,init_proba=0.0,step_floor=10000,signal=False):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """

        # Bandit_n refers to the single players

        super(EpsilonGreedy_auction, self).__init__(bandit_n)

        assert 0. <= eps <= 1.0
        self.eps = eps
        self.t=0
        self.start_point=start_point # Number of exploration steps

        self.bidding_range=bidding_range
        self.overbid=overbid
        self.signal=signal

        self.true_value_list=[]

        self.decay = 0.5 #3 * self.bandit_n #ori: 10
        self.step_floor=step_floor # Number of steps to eval the avg. maybe eval_step_size
        self.estimates = [init_proba] * self.bandit_n  # Optimistic initialization
        # list len = n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.eps:
            # Let's do random exploration!
            i = [np.random.randint(0, bandit_n) for bandit_n in self.bandit_ns]  # ***
        else:
            # Pick the best one.
            i = max(range(self.bandit_n), key=lambda x: self.estimates[x])

        # get reward
        r = self.bandit.generate_reward(i)
        #

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i

    # eric

    def generate_action(self,obs=None,test=False,upper_bound=None):
        # Obs refers to the true value

        # Lower bid does not refer to the actual lower bid, but the lower bid on the 
        # encoded space.
        if upper_bound is not None:
            lower_bid = obs * self.bidding_range
            upper_bid = obs*self.bidding_range + upper_bound+1

            #print(lower_bid)
            #print(upper_bid)


        elif self.overbid or self.signal:
            lower_bid = obs * self.bidding_range
            upper_bid = (obs+1)*self.bidding_range
        else:
            lower_bid=obs*self.bidding_range
            upper_bid =obs*self.bidding_range+obs+1     #  over bid: (obs+1)*self.bidding_range

        if (np.random.random() < self.eps or self.t<self.start_point ) and (test==False):
            # Let's do random exploration!
            #i = np.random.randint(0, self.bandit_n)
            # updated to the m

            if obs is None:
                action_possibility = np.array([ 1/(num+1e-6) for num in self.counts])
                action_possibility /=  sum(action_possibility)
                i = np.random.choice([i for i in range(self.bandit_n)], p=action_possibility.ravel())
            else:
                # allow over bid
                i = np.random.choice([i for i in range(lower_bid,upper_bid)])
                # forbid over bid :


            #print('rnd!!!!!!!!')
        elif obs is not None:
            # Pick the best one.
            #i = max(range(self.bandit_n), key=lambda x: self.estimates[x])

            # with shuffle
            select_estimate=self.estimates[lower_bid : upper_bid]
            max_value = max(select_estimate)

            max_act_list = [k for k, v in enumerate(select_estimate) if v == max_value]
            random.shuffle(max_act_list)

            i=max_act_list[0] +lower_bid #fixed bug in 11.3
            # print(i-lower_bid)
            # print(i)
            # print('---')
        else:
            max_act_list = [k for k, v in enumerate(self.estimates) if v == max(self.estimates)]
            random.shuffle(max_act_list)
            i = max_act_list[0]
        # record
        if not test:

            self.record_action(i)

        return i

    def update_policy(self, i,reward):  # do  record_action(self,i) before update policy

        # self.estimates[i] += 1. / (self.counts[i] + 1) * (self.reward - self.estimates[i])
        # old version as cnt+1 after update policy
        #print('at round' + str(self.t) +'  optimize is ' + str( (self.reward - self.estimates[i])) )
        #if (self.reward - self.estimates[i]) !=0:
            #print('weighted is :' + str(1. / (self.counts[i] + 1)))

        #self.estimates[i] += 1. / (self.counts[i] + 1) * (self.reward - self.estimates[i])

        self.estimates[i] = (reward +self.estimates[i] * self.counts[i]) / (self.counts[i]+1)

        #self.estimates[i] = (self.reward + self.estimates[i]) /2

        #self.estimates[i] = (self.reward *0.1+  0.9 * self.estimates[i])


        self.t+=1

        if self.t ==self.start_point :
            # remove the accumalative counts for optimization
            self.counts=[1] * self.bandit_n


        self.eps= 0.5 **(self.t /  self.step_floor)


class UCB1(Solver):
    def __init__(self, bandit_n, init_proba=1.0):
        super(UCB1, self).__init__(bandit_n)
        self.t = 0
        self.estimates = [init_proba] * self.bandit_n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # Pick the best one with consideration of upper confidence bounds.
        i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.t) / (1 + self.counts[x])))

        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i

    # eric
    def generate_action(self):
        self.t+=1
        i = max(range(self.bandit_n), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.t) / (1 + self.counts[x])))
        return i

    def update_policy(self, i):  # do  record_action(self,i) before update policy

        # self.estimates[i] += 1. / (self.counts[i] + 1) * (self.reward - self.estimates[i])
        # old version as cnt+1 after update policy
        #print(self.reward)

        self.estimates[i] += 1. / (self.counts[i]) * (self.reward - self.estimates[i])



class UCB1(Solver):
    def __init__(self, bandit_n, init_proba=1.0):
        super(UCB1, self).__init__(bandit_n)
        self.t = 0
        self.estimates = [init_proba] * self.bandit_n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # Pick the best one with consideration of upper confidence bounds.
        i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.t) / (1 + self.counts[x])))

        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i

    # eric
    def generate_action(self):
        self.t+=1
        i = max(range(self.bandit_n), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.t) / (1 + self.counts[x])))
        return i

    def update_policy(self, i):  # do  record_action(self,i) before update policy

        # self.estimates[i] += 1. / (self.counts[i] + 1) * (self.reward - self.estimates[i])
        # old version as cnt+1 after update policy
        #print(self.reward)

        self.estimates[i] += 1. / (self.counts[i]) * (self.reward - self.estimates[i])




class BayesianUCB(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit_n, std_dev=3, Beta_a=1, Beta_b=1):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(BayesianUCB, self).__init__(bandit_n)
        self.c = std_dev
        self._as = [Beta_a] * self.bandit_n
        self._bs = [Beta_b] * self.bandit_n

    @property
    def estimated_probas(self):
        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit_n)]

    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        i = max(
            range(self.bandit_n),
            key=lambda x: self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(
                self._as[x], self._bs[x]) * self.c
        )
        r = self.bandit.generate_reward(i)

        # Update Gaussian posterior
        self._as[i] += r
        self._bs[i] += (1 - r)

        return i

    # eric
    def generate_action(self):
        i = max(
            range(self.bandit_n),
            key=lambda x: self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(
                self._as[x], self._bs[x]) * self.c
        )

        return i

    def update_policy(self, i):  # do  record_action(self,i) before update policy

        # self.estimates[i] += 1. / (self.counts[i] + 1) * (self.reward - self.estimates[i])
        # old version as cnt+1 after update policy
        self._as[i] += self.reward
        self._bs[i] += (1 - self.reward)


class ThompsonSampling(Solver):
    def __init__(self, bandit_n, Beta_a=1, Beta_b=1):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(ThompsonSampling, self).__init__(bandit_n)

        self._as = [Beta_a] * self.bandit_n
        self._bs = [Beta_b] * self.bandit_n

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit_n)]

    def run_one_step(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit_n)]
        i = max(range(self.bandit_n), key=lambda x: samples[x])
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1 - r)

        return i

    # eric
    def generate_action(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit_n)]
        i = max(range(self.bandit_n), key=lambda x: samples[x])

        return i

    def update_policy(self, i):  # do  record_action(self,i) before update policy

        # self.estimates[i] += 1. / (self.counts[i] + 1) * (self.reward - self.estimates[i])
        # old version as cnt+1 after update policy
        self._as[i] += self.reward
        self._bs[i] += (1 - self.reward)

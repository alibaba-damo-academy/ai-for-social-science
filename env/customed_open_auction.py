from gymnasium.spaces import Discrete
import numpy as np
import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
import random
from copy import deepcopy

class open_auction(AECEnv):
    """
    The metadata holds environment constants. From gym, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "second price"}

    def __init__(self,player_num=2,action_space_num = 13,policy=None,reserve_price=0,max_bidding_times=1000,reveal_all_bid=False):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """

        self.player_num=player_num
        self.action_space_num=action_space_num
        self.env_iters=max_bidding_times
        self.reserve_price=reserve_price

        self.policy=policy
        self.render_mode="human"

        self.last_high_bid=0
        self.cur_high_bid=0
        self.last_winner=None
        self.last_state=None
        self.reveal_all_bid=reveal_all_bid





        self.possible_agents = ["player_" + str(r) for r in range(player_num)]


        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self._action_spaces = {agent: Discrete(action_space_num) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Discrete(action_space_num+1) for agent in self.possible_agents
        }

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces


        #will modified in the future

        return Discrete(self.action_space_num+1)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.action_space_num)

    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if len(self.agents) == self.player_num:
            string = "Current state:"
            cnt=0
            for agt in self.possible_agents:
                string +="Agent {} bid price with: {} ,".format(
                cnt+1, self.state[agt]
            )
                cnt+=1
        else:
            string = "Game over"
        print(string)


    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        # res=[]
        # for agt in self.possible_agents:
        #     if agt ==agent:
        #         continue
        #     else:
        #         res.append(self.observations[agt])

        #observation in sealed auction is whether he wins the auction
        return {"observation": self.observations[agent] }

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None,options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        NONE = self.action_space_num +1

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: NONE for agent in self.agents} # The default value is given as the action space + 1
        self.num_moves = 0

        self.full_state={agent: NONE for agent in self.agents}

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations (termination condition of the agent)
        - truncations (Truncate if running beyond specified rounds)
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection] #if not bid anymore -> terminations
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_dead_step(action)

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action


        # check if all the agent submit their bid
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary




            if self.policy is not None:
                winner = self.policy.compute_allocation(self.state,reserve_price=self.reserve_price)
                #print(winner)
                self.rewards = self.policy.compute_payment(self.state,winner=winner)
                #print(1)

            else:
                print(
                    'no find winner : assign for random and pay with highest'
                )
                winner = self.possible_agents[random.randint(0,len(self.possible_agents))]
                for agt in self.agents:
                    self.rewards[agt]=0
                self.rewards[winner] = self.state[winner]

            # get the current high bid
            if winner is not None:

                self.cur_high_bid =self.state[winner] #max(self.state)
            else:
                # all bid lower than the reserve price
                self.cur_high_bid = self.reserve_price


            self.num_moves += 1
            # The truncation dictionary must be updated for all players.
            self.truncations = {agent: self.num_moves >= self.env_iters for agent in self.agents}

            # should consider if the auction whens to the maximal bidding time
            # if the auction whens to the end:

            if self.num_moves >= self.env_iters or self.state == self.last_state:
                #or (self.cur_high_bid>self.reserve_price and self.cur_high_bid == self.last_high_bid and self.last_winner == winner) \
                 #self.num_moves >= self.env_iters or self.state == self.last_state:

                # not more higher bid
                # only one bidders bid reach the maximal bidding limit

                # finish the terminations
                self.terminations = {agent: 1 for agent in self.agents}
                #
                # print('-------')
                # print('current state is ')
                # print(self.state)
                # print('last state is ')
                # print(self.last_state)
                # print('-------')
                #compute the final allocation and the reward


                # assign the final allocation
                for i in self.agents:
                    self.infos[i]['allocation'] = 0
                if winner is not None:
                    # at least higher than reserve price
                    self.infos[winner]['allocation']=1



            else:
                # observe the current state when auction is not end
                for i in self.agents:
                    self.infos[i]['allocation'] = -1 # not finish
                    self.infos[i]['highest_bid'] = self.cur_high_bid #report the current highest bid

                    if self.reveal_all_bid:
                        self.infos[i]['all_bid'] = self.state

                    if i == winner:  # in winner_list: #== winner :
                        self.observations[i] = 1
                        self.infos[i]['tmp_allocation'] = 1
                        self.infos[i]['tmp_reward'] = self.rewards[i]

                        # self.infos[i]['other_value'] = self.policy.get_other_value(agent=i)
                        # self.infos[i]['cooperate_win'], self.infos[i][
                        #     'cooperate_pay'] = self.policy.check_cooperate_win(agent=i)

                    else:
                        self.observations[i] = 0
                        self.infos[i]['tmp_allocation'] = 0
                        self.infos[i]['tmp_reward'] = 0

                        # self.infos[i]['other_value'] = self.policy.get_other_value(agent=i)
                        # self.infos[i]['cooperate_win'], self.infos[i][
                        #     'cooperate_pay'] = self.policy.check_cooperate_win(agent=i)

            #record this round
            self.last_state=deepcopy(self.state)


            self.last_high_bid=deepcopy(self.cur_high_bid)
            self.last_winner=deepcopy(winner)

        else:
            # necessary so that observe() returns a reasonable observation at all times.
            # if self.last_winner is not None and agent == self.last_winner and action < self.last_high_bid:
            #         self.state[agent] =self.last_high_bid

            if action < self.reserve_price:
                # mark -1 as the no effect bid
                self.state[agent] = -1
            else:
                self.state[agent] = action

            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

if __name__ == '__main__':
    print(1)
    # import sys
    #
    #
    # sys.path.append('../')
    # from env.policy import *
    #
    # # Initialize the auction allocation/payment policy of the auctioneer
    # test_policy = Policy(allocation_mode='highest', payment_mode='second_price',\
    #                     )
    #
    # env = open_auction(player_num=3,
    #                         action_space_num=10,
    #                         max_bidding_times=100,
    #                         policy=test_policy
    #                         )
    # # This wrapper is only for environments which print results to the terminal
    # env = wrappers.CaptureStdoutWrapper(env)
    # # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # # Provides a wide vareity of helpful user errors
    # # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)
    #
    # env.reset()
    # print(env.agents)
    #
    # # print(args.inner_cooperate_id)
    #
    # test_policy.assign_agent(env.agents)
    # agt_list = {agent: None for agent in env.agents}
    #
    # epoch=0
    #
    # for agent_name in env.agent_iter():
    #     epoch+=1
    #     #print('cur epoch is ' + str(epoch))
    #
    #     observation, reward, termination, truncation, info = env.last()
    #     _obs = observation['observation']
    #
    #     if _obs == 10 + 1 :
    #         print('first_round')
    #         highest_bid=0
    #         tmp_allocation=0
    #     else:
    #         highest_bid = info['highest_bid']
    #
    #         if termination :
    #             print('end of the auction of agent ' + str(agent_name))
    #
    #             final_pay = reward
    #             allocation = info['allocation']
    #             if allocation==1:
    #                 print(agent_name + 'final payment is '+str(final_pay) + ' | his true value is ' + str(int(agent_name[-1])+2) )
    #
    #         tmp_allocation = info['tmp_allocation']
    #
    #     # submit bid
    #
    #     true_value = int(agent_name[-1])+2
    #     #generate action
    #     if tmp_allocation:
    #         bid = highest_bid #last bid
    #
    #     elif highest_bid<true_value:
    #         bid= highest_bid+1
    #     else:
    #         bid = true_value
    #
    #     if termination or truncation:
    #         env.step(None)
    #     else:
    #         print(agent_name + ' submit bid '+str(bid) + ' with his true value ' + str(true_value) + 'within epoch' + str(epoch))
    #         env.step(bid)

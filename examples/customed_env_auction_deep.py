#from gym.spaces import Discrete
from gymnasium.spaces import Discrete
import numpy as np
import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
import random


#env_iters =5#000000 #80000000



def customed_env(player_num=2,action_space_num=10,second_price=True,env_iters =5):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    if second_price==True:
        env = second_env_auction(player_num=player_num,action_space_num=action_space_num,env_iters =env_iters)
    else:
        env = first_env_auction(player_num=player_num, action_space_num=action_space_num)
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class second_env_auction(AECEnv):
    """
    The metadata holds environment constants. From gym, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "second price"}

    def __init__(self,player_num=2,action_space_num = 13,env_iters =5):
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
        self.env_iters=env_iters
        self.render_mode = "human"
        self.possible_agents = ["player_" + str(r) for r in range(player_num)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.action_spaces = {agent: Discrete(action_space_num) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Discrete(action_space_num+1) for agent in self.possible_agents
        }

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

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

        return self.observations[agent]

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
        - terminations (termination condition of the agent)
        - truncations (Truncate if running beyond specified rounds)
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        NONE = self.action_space_num

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: 0 for agent in self.agents}
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
            self.terminations[self.agent_selection]
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

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary


            # first price
            # second price
            winner_list=[]
            highest_bid=0
            for agt in self.state:
                highest_bid = max(self.state[agt],highest_bid)

            second_highest_bid = 0

            for agt in self.possible_agents:
                agt_bid = self.state[agt]
                if agt_bid == highest_bid:
                    winner_list.append(agt)
                else:
                    second_highest_bid=max(second_highest_bid,agt_bid)

            if len(winner_list)>1:
                second_highest_bid=highest_bid

            # multi-winner situation

            # only one winner
            if len(winner_list)>1:
                random.shuffle(winner_list)
            winner=winner_list[0]



            # self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
            #     (self.state[self.agents[0]], self.state[self.agents[1]])
            # ]

            self.num_moves += 1
            # The truncation dictionary must be updated for all players.
            self.truncations = {agent: self.num_moves >= self.env_iters for agent in self.agents}

            # observe the current state
            for i in self.agents:

                if i ==winner : #in winner_list: #== winner :

                    self.rewards[i] = (self.observations[i] - second_highest_bid) *1.0 / self.action_space_num
                    self.infos[i]['allocation'] = 1
                    # observation are the next round true value
                    self.observations[i] = random.randint(0,self.action_space_num-1) #【0，action_space-1]

                else:
                    self.rewards[i]=0
                    self.infos[i]['allocation'] = 0
                    self.observations[i] = random.randint(0,self.action_space_num-1) #【0，action_space-1]


        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[agent] = action

            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

class first_env_auction(AECEnv):
    """
    The metadata holds environment constants. From gym, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "second price"}

    def __init__(self,player_num=2,action_space_num = 13,env_iters =5):
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
        self.env_iters=env_iters
        self.render_mode = "human"


        self.possible_agents = ["player_" + str(r) for r in range(player_num)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.action_spaces = {agent: Discrete(action_space_num) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Discrete(action_space_num+1) for agent in self.possible_agents
        }

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

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

        return self.observations[agent]

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
        - terminations (termination condition of the agent)
        - truncations (Truncate if running beyond specified rounds)
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        NONE = self.action_space_num

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: 0 for agent in self.agents}
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
            self.terminations[self.agent_selection]
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

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary


            # first price
            # second price
            winner_list=[]
            highest_bid=0
            for agt in self.state:
                highest_bid = max(self.state[agt],highest_bid)

            second_highest_bid = 0

            for agt in self.possible_agents:
                agt_bid = self.state[agt]
                if agt_bid == highest_bid:
                    winner_list.append(agt)
                else:
                    second_highest_bid=max(second_highest_bid,agt_bid)

            if len(winner_list)>1:
                second_highest_bid=highest_bid

            # multi-winner situation

            # only one winner
            if len(winner_list)>1:
                random.shuffle(winner_list)
            winner=winner_list[0]



            # self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
            #     (self.state[self.agents[0]], self.state[self.agents[1]])
            # ]

            self.num_moves += 1
            # The truncation dictionary must be updated for all players.
            self.truncations = {agent: self.num_moves >= self.env_iters for agent in self.agents}

            # observe the current state
            for i in self.agents:

                if i ==winner : #in winner_list: #== winner :

                    self.rewards[i] = (self.observations[i] - highest_bid) *1.0 / self.action_space_num
                    self.infos[i]['allocation'] = 1
                    # observation are the next round true value
                    self.observations[i] = random.randint(0,self.action_space_num-1) #【0，action_space-1]

                else:
                    self.rewards[i]=0
                    self.infos[i]['allocation'] = 0
                    self.observations[i] = random.randint(0,self.action_space_num-1) #【0，action_space-1]


        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[agent] = action

            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


from gym.spaces import Discrete,MultiDiscrete
import numpy as np
import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from env.policy import *
import random

def multi_item_static_env(args,policy=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    if args.action_mode =='div':
        env = fixed_env_auction(player_num=args.player_num,
                            action_space_num=args.bidding_range * args.valuation_range,
                            env_iters=args.env_iters,policy=policy
                                )

    else:
        env = multi_static_env(player_num=args.player_num,
                            action_space_num=args.bidding_range,
                            env_iters=args.env_iters,policy=policy,
                                )



    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class multi_static_env(AECEnv):
    """
    The metadata holds environment constants. From gym, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "second price"}

    def __init__(self,player_num=2,action_space_num = 13,item_num=5, \
        env_iters=50000,policy=None):
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
        self.item_num=item_num

        self.policy=policy
        self.render_mode = "human"


        self.possible_agents = ["player_" + str(r) for r in range(player_num)]

        self.auctioned_items=["item_"+ str(r) for r in range(item_num)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.item_name_mapping = dict(
            zip(self.auctioned_items, list(range(len(self.auctioned_items))))
        )

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self._action_spaces = {agent: MultiDiscrete([action_space_num]*item_num) for agent in self.possible_agents}
        # or change to diversed action space
        # multi discrete  e.g [5,3,3]

        self._observation_spaces = {
            agent: MultiDiscrete([action_space_num+1]*item_num) for agent in self.possible_agents
        }

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return MultiDiscrete([self.action_space_num+1]*self.item_num) #Discrete(self.action_space_num+1)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return MultiDiscrete([self.action_space_num]*self.item_num) #Discrete(self.action_space_num)

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
        - terminations (termination condition of the agent)
        - truncations (Truncate if running beyond specified rounds)
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        NONE = self.action_space_num +1

        self.agents = self.possible_agents[:]
        # version : sum reward or diversed
        self.rewards = {agent: 0 for agent in self.agents}
        self.discrete_reward={agent: [NONE]*self.item_num for agent in self.agents}
        #
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: [NONE]*self.item_num for agent in self.agents}
        self.mini_state ={agent: NONE for agent in self.agents}

        self.observations = {agent: [NONE]*self.item_num for agent in self.agents}
        self.num_moves = 0

        self.full_state={agent: [NONE]*self.item_num for agent in self.agents}

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
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
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

            if self.policy.args.mechanism == 'vcg':
                current_winners = self.policy.compute_allocation(self.state) 
                payment = self.policy.compute_payment(self.state)
                
                for tmp_agt in self.agents:
                    self.observations[tmp_agt] =  [int(winner== tmp_agt) for winner in current_winners]
                    self.infos[tmp_agt]['allocation'] = self.observations[tmp_agt] # total allocation
                    self.rewards[tmp_agt] = payment[tmp_agt]
                
            elif self.policy is not None:
                #apply multi item auction
                for item_id in range(self.item_num):
                    current_item_name = self.auctioned_items[item_id]
                    for tmp_agt in self.agents:
                        self.mini_state[tmp_agt]=self.state[tmp_agt][item_id]
                    current_winner = self.policy.compute_allocation(self.mini_state) 
                    current_rewards=self.policy.compute_payment(self.mini_state,winner=current_winner)
                    for tmp_agt in self.agents:
                        self.discrete_reward[tmp_agt][item_id] = current_rewards[tmp_agt]
                        self.observations[tmp_agt][item_id] = 0 #init

                    self.observations[current_winner][item_id] = 1

                for tmp_agt in self.agents:
                    self.infos[tmp_agt]['allocation']= self.observations[tmp_agt] # total allocation
                    self.rewards[tmp_agt]=sum(
                        self.discrete_reward[tmp_agt]
                    ) 
                    num_items_win = sum(self.observations[tmp_agt]) 
                    if num_items_win > 1:
                        self.rewards[tmp_agt] *= self.policy.args.multi_item_decay**num_items_win

                #winner = self.policy.compute_allocation(self.state)
                #print(winner)
                #self.rewards = self.policy.compute_payment(self.state,winner=winner)
                #print(1)

            else:
                print(
                    'no find winner : assign for random and pay with highest'
                )
                winner = self.possible_agents[random.randint(0,len(self.possible_agents))]
                for agt in self.agents:
                    self.rewards[agt]=0
                self.rewards[winner] = self.state[winner]

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.truncations = {agent: self.num_moves >= self.env_iters for agent in self.agents}

            for i in self.agents: 
                self.infos[i]['other_value']=None
            
            # # observe the current state
            # for i in self.agents:
            #
            #     if i ==winner : #in winner_list: #== winner :
            #         self.observations[i] = 1
            #         self.infos[i]['allocation'] = 1
            #
            #         self.infos[i]['other_value']=self.policy.get_other_value(agent=i)
            #         self.infos[i]['cooperate_win'],self.infos[i]['cooperate_pay'] = self.policy.check_cooperate_win(agent=i)
            #
            #     else:
            #         self.observations[i]=0
            #         self.infos[i]['allocation'] = 0
            #         self.infos[i]['other_value'] = self.policy.get_other_value(agent=i)
            #         self.infos[i]['cooperate_win'],self.infos[i]['cooperate_pay'] = self.policy.check_cooperate_win(agent=i)

        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[agent] = action

            # no rewards are allocated until both players give an action
            self._clear_rewards()   
        
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


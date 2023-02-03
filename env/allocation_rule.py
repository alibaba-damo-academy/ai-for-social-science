import random
from agent_utils.action_encoding import *
import itertools
import random 


def highest_allocation(state,possible_agents):
    """Return the agents with the highest state"""

    winner_list = []
    highest_bid = 0
    for agt in state:
        highest_bid = max(state[agt], highest_bid)

    second_highest_bid = 0

    for agt in possible_agents:
        agt_bid = state[agt]
        if agt_bid == highest_bid:
            winner_list.append(agt)
        else:
            second_highest_bid = max(second_highest_bid, agt_bid)

    if len(winner_list) > 1:
        second_highest_bid = highest_bid

    # multi-winner situation

    # only one winner
    if len(winner_list) > 1:
        random.shuffle(winner_list)
    winner = winner_list[0]

    return winner


# VCG solver 

# Consider the simplest case where each player place bid only for individual 
# item, and for any item pair, they are willing to pay the sum of individual bid

# Example: 3 items, 3 player 
# Player 1: 1, 4, 5 
# Player 2: 2, 6, 3 
# Player 3, 4, 5, 2  

# Then, the VCG payment is (4+6+3) - (4+6+5-5); (4+5+5) - (4+6+5-6); (2+6+5) - (4+6+5-4)
# i.e. multi-round second price auction. 


# Total welfare is given as the sum of the revenue. Then, the optimal allocation  
# can be found by finding the sum. 

# Then, we need to find payment for individual bidder
    
def get_vcg_welfare(bids, possible_agents, winners, decay): 
    # Compute the VCG welfare given decay.
    
    welfares = {agent: [] for agent in possible_agents} 
    for i, winner in enumerate(winners):
        welfares[winner].append(bids[winner][i])
    welfare = 0
    for v in welfares.values():  
        if len(v) > 0: 
            welfare += sum(v)*decay**(len(v)-1)
    return welfare

def vcg_allocation(bids,possible_agents, decay = 1):
    print(bids,possible_agents)
    # Use a different variable than state to specify the bids e.g. 
    #  {"agent1": [1,4,5], "agent2": [2,6,3], "agent3": [4,5,2]}  
    
    
    # check valid bids: length is the same for all agents.
    
    # In multi-item allocation case, we need to specify the  
    winners =  [] # length same as the item.
    winners_welfare = 0
    
    item_num = len(bids['player_0'])
    
    # Todo: improve with DP 
    for possible_winners in itertools.product(*[possible_agents]*item_num): 
        possible_winners = list(possible_winners)
        possible_welfare = get_vcg_welfare(bids, possible_agents, possible_winners, decay) 
        if possible_welfare > winners_welfare: 
            winners_welfare = possible_welfare 
            winners = possible_winners
    return winners, winners_welfare

def topk_allocation(bids,possible_agents):
    # Allocate the identical items to the agent 
    
    item_num = len(bids['player_0'])
    bids_flatten = []
    for agent, bid in bids.items(): 
        assert len(bid) == item_num, "bidder {} should provide {} bids!".format(agent, item_num)
        bids_flatten = bids_flatten + [(b, agent) for b in bid]
    
    winners =  {agent: 0 for agent in possible_agents} # length same as the agent. (each value refers to the number of units won)
    bids_flatten.sort(key = lambda x: x[0],reverse=True)
    for (_, agent) in bids_flatten[:item_num]: 
        winners[agent] += 1
    sorted_bids = bids_flatten
    return winners, sorted_bids

if __name__ == "__main__": 
    bids = {'player_0': [2, 3, 1, 7, 4], 'player_1': [0, 0, 0, 0, 0], 'player_2': [5, 8, 7, 3, 7], 'player_3': [0, 9, 5, 8, 0], 'player_4': [6, 3, 7, 0, 1]}
    possible_agents = ['player_0', 'player_1', 'player_2', 'player_3', 'player_4']

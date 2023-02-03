from agent_utils.action_encoding import *
import copy
import numpy as np
from .allocation_rule import *


# We can probably define a class of payment rule

def pay_by_submit(bids,possible_agents,winner):
    """
    All agents pay the bids regardless of winning/losing.
    """
    payment = {agent: 0 for agent in possible_agents}

    for i in possible_agents:
        payment[i] = -1 * bids[i]

    return payment

# We can combine the following functions later

def third_price_rule(bids,possible_agents,winner):

    payment={agent: 0 for agent in possible_agents}

    third_highest_bid = get_rank(bids,possible_agents,rank=3)

    for i in possible_agents:

        if i == winner:  # in winner_list: #== winner :

            payment[i] = -1 * third_highest_bid

    return payment

def second_price_rule(bids,possible_agents,winner):

    payment={agent: 0 for agent in possible_agents}

    second_highest_bid = get_rank(bids,possible_agents,rank=2)

    for i in possible_agents:

        if i == winner:  # in winner_list: #== winner :

            payment[i] = -1 * second_highest_bid

    return payment
def first_price_rule(bids,possible_agents,winner):
    payment={agent: 0 for agent in possible_agents}

    first_highest_bid = get_rank(bids,possible_agents,rank=1)

    for i in possible_agents:

        if i == winner:  # in winner_list: #== winner :

            payment[i] = -1 * first_highest_bid

    return payment
def get_rank(bids,possible_agents,rank=2):

    data=[]
    for agt in possible_agents:
        data.append(bids[agt])

    data.sort()

    if(rank)>len(data):
        return data[0]
    else:
        return data[-rank]


def vcg_payment(bids, possible_agents, decay=0.9): 
    payment={agent: 0 for agent in possible_agents}
    
    winners, welfare = vcg_allocation(bids, possible_agents, decay)
    # Revenue is equivalent to welfare in this case
    for i in possible_agents:
        item_wins = [int(winner== i) for winner in winners] # [1, 0, 0] 
        partial_welfare = 0
        if sum(item_wins) > 0: 
            partial_welfare = np.dot(item_wins, bids[i])*decay**(sum(item_wins)-1)
        sub_bids = copy.deepcopy(bids)
        sub_bids[i] = [0 for _ in sub_bids[i]] 
        # print(i)
        # print("sub_bids", sub_bids)
        sub_winners, sub_welfare = vcg_allocation(sub_bids, possible_agents, decay) 
        # print("sub_win",sub_winners)
        
        # sub_welfare refers to the total welfare of the subgame
        # partial_welfare refers to the partial welfare of the orignal game by agent i
        payment[i] = sub_welfare - (welfare - partial_welfare)
        # print("sub", sub_welfare, "welfare", welfare,"partial", partial_welfare)
        # print(i, payment) 
        payment[i] = -1* payment[i] # Payment is negative, reward is positive
    return payment



def discriminatory_multi_payment(bids, possible_agents): 
    # Pay your submit
    payment = {agent: 0 for agent in possible_agents}
    item_num = len(bids['player_0'])
    _, sorted_bids = topk_allocation(bids,possible_agents)
    for (bid, agent) in sorted_bids[:item_num]:
        payment[agent] += bid
    return payment

def uniform_multi_payment(bids, possible_agents): 
    # Highest losing bid
    item_num = len(bids['player_0'])
    _, sorted_bids = topk_allocation(bids,possible_agents)
    return {agent: sorted_bids[item_num][0] for agent in possible_agents}
    
def Vickrey_multi_payment(bids, possible_agents): 
    # Pay the k th highest losing bid not including his bid
    # TODO
    return

if __name__ == "__main__":
    possible_agents = {"agent1", "agent2", "agent3"}
    bids = {"agent1": [1,4,5], "agent2": [2,6,3], "agent3": [4,5,2]}  
    # winners = ["agent3", "agent2", "agent1"] 
    # welfare = get_vcg_welfare(bids, possible_agents, winners, decay = 1)
    # print(vcg_allocation(bids,possible_agents, decay = 0.9))
    print(uniform_multi_payment(bids, possible_agents))
import random
from env.allocation_rule import *
from env.payment_rule import *
from agent_utils.action_encoding import *

class stackelberg_Policy(object):
    """
    Allocation policy that decides which agent to allocate the auction item to
    and the payment each agent pays to the auctioneer
    """
    def __init__(self, allocation_mode='highest', payment_mode='second_price',action_mode='same',args=None):
        self.possible_agents = None
        self.state = None
        self.allocation_mode = allocation_mode
        self.payment_mode = payment_mode
        self.action_mode=action_mode
        self.args=args


        self.last_winner=None
        self.last_payment=None

    def assign_agent(self, agents):
        self.possible_agents = agents
        self.info={agent: None for agent in self.possible_agents}


    def assign_state(self, state):
        self.state = state

    def assign_payment_mode(self, payment_mode):
        self.payment_mode = payment_mode

    def assign_allocation_mode(self,allocation_mode):
        self.allocation_mode=allocation_mode

    def compute_allocation(self, state,reserve_price=0):
        if self.action_mode=='div':
            state = de_transform_state(state=state, args=self.args)
        if self.allocation_mode == 'highest':
            winner = highest_allocation(state=state, possible_agents=self.possible_agents,reserve_price=reserve_price)
        if self.allocation_mode == 'lowest':
            winner = lowest_allocation_mod(state=state, possible_agents=self.possible_agents,reserve_price=reserve_price)


        if self.allocation_mode == 'vcg':
            winner, _ = vcg_allocation(state, possible_agents = self.possible_agents,
                                    decay = self.args.multi_item_decay)
        self.last_winner=winner


        return winner

    def compute_payment(self, state, winner=None):


        if self.action_mode=='div':
            state= de_transform_state(state=state,args=self.args)

        if self.payment_mode == 'second_price':
            payment = second_price_rule(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'first_price':
            payment = first_price_rule(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode=='lowest':
            payment =lowest_payment(state, possible_agents=self.possible_agents, winner=winner)


        elif self.payment_mode =='third_price':
            payment = third_price_rule(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode =='pay_by_submit':
            payment = pay_by_submit(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'vcg':
            payment = vcg_payment(state, possible_agents=self.possible_agents, decay = self.args.multi_item_decay)
        else:
            print('not find the payment mode' + str(self.payment_mode))

        self.last_payment=payment

        return payment


    def set_information(self,agent,infos=None):

        self.info[agent]=infos
        return
    def set_information_by_id(self,agent_id,infos=None):

        agent=self.possible_agents[agent_id]
        self.info[agent]=infos
        return

    def get_info(self,agent):
        return self.info[agent]

    def get_info_by_id(self,agent_id):

        agent=self.possible_agents[agent_id]
        return self.info[agent]

    ## cooperate
    def get_other_value(self,agent):

        sum_value=None
        if self.args.inner_cooperate==0:
            return None


        agent_id = self.possible_agents.index(agent)




        if agent_id not in self.args.inner_cooperate_id: # not in inner cooperate id
            return None

        for id in self.args.inner_cooperate_id:
            infos = self.get_info_by_id(agent_id=id)
            if id !=agent_id and (infos is not None):

                if sum_value is None:
                    sum_value=infos
                else:
                    sum_value+=infos


        return sum_value

    def check_cooperate_win(self,agent):
        if self.args.inner_cooperate == 0:
            return None,None
        else:
            agent_id = self.possible_agents.index(agent)
            if agent_id not in self.args.inner_cooperate_id:  # not in inner cooperate id
                return None,None

            else:
                winner_idx = self.possible_agents.index(self.last_winner)
                if winner_idx in self.args.inner_cooperate_id:

                    return True, self.last_payment[self.last_winner]
                else:
                    return False,None



def lowest_payment(bids,possible_agents,winner):
    payment = {agent: 0 for agent in possible_agents}

    for i in possible_agents:

        if i == winner:  # in winner_list: #== winner :

            payment[i] = bids[winner]

    return payment


def lowest_allocation_mod(state, possible_agents, reserve_price):
    lowest_bid = min(state.values())
    winner_list = []
    for agt in possible_agents:
        agt_bid = state[agt]
        if agt_bid == lowest_bid:
            winner_list.append(agt)
    # only one winner
    if len(winner_list) > 1:
        random.shuffle(winner_list)
    winner = winner_list[0]
    return winner
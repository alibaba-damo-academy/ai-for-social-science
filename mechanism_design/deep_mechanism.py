import torch

import sys
sys.path.append('../')

from env.policy import *

from copy import deepcopy

class deep_policy(Policy):

    def __init__(self,allocation_mode='deep', payment_mode='deep',action_mode='same',args=None):
        super(deep_policy).__init__()

        self.possible_agents = None
        self.state = None
        self.allocation_mode = allocation_mode
        self.payment_mode = payment_mode
        self.action_mode = action_mode
        self.args = args

        self.last_winner = None
        self.last_payment = None

        self.allocation_net = None
        self.payment_net = None

    def compute_allocation(self, state,reserve_price=0):
        if self.action_mode=='div':
            state = de_transform_state(state=state, args=self.args)
        if self.allocation_mode == 'highest':
            winner = highest_allocation(state=state, possible_agents=self.possible_agents,reserve_price=reserve_price)

        elif self.allocation_mode == 'deep':
            winner = self.allocation_net.compute_allocation(state=state, possible_agents=self.possible_agents,reserve_price=reserve_price)


        self.last_winner=winner


        return winner

    def compute_payment(self, state, winner=None):

        if self.action_mode == 'div':
            state = de_transform_state(state=state, args=self.args)

        if self.payment_mode == 'second_price':
            payment = second_price_rule(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'first_price':
            payment = first_price_rule(state, possible_agents=self.possible_agents, winner=winner)

        elif self.payment_mode == 'third_price':
            payment = third_price_rule(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'pay_by_submit':
            payment = pay_by_submit(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'deep':
            # add deep net payment
            payment = self.payment_net.compute_payment(state, possible_agents=self.possible_agents, winner=winner)

        else:
            print('not find the payment mode' + str(self.payment_mode))

        self.last_payment = payment

        return payment

    def assign_payment_net(self,payment_net):
        self.payment_net=payment_net

    def assign_allocation_net(self,allocation_net):
        self.allocation_net = allocation_net

    def export_payment_net(self):
        return deepcopy(self.payment_net)

    def export_allocation_net(self):
        return deepcopy(self.allocation_net)


    def close_grad(self):
        if self.payment_mode== 'deep' and self.payment_net is not None:
            self.payment_net.close_grad()

        if self.allocation_mode== 'deep' and self.allocation_net is not None:
            self.allocation_net.close_grad()


    def open_grad(self):

            if self.payment_mode == 'deep' and self.payment_net is not None:
                self.payment_net.open_grad()

            if self.allocation_mode == 'deep' and self.allocation_net is not None:
                self.allocation_net.open_grad()

#
# tmp =deep_policy(allocation_mode='deep')
# agents=['agent_1','agent_2']
# tmp.assign_agent(agents)
# print(tmp)


class deep_policy(Policy):

    def __init__(self,allocation_mode='deep', payment_mode='deep',action_mode='same',args=None):
        super(deep_policy).__init__()

        self.possible_agents = None
        self.state = None
        self.allocation_mode = allocation_mode
        self.payment_mode = payment_mode
        self.action_mode = action_mode
        self.args = args

        self.last_winner = None
        self.last_payment = None

        self.allocation_net = None
        self.payment_net = None

    def compute_allocation(self, state,reserve_price=0):
        if self.action_mode=='div':
            state = de_transform_state(state=state, args=self.args)
        if self.allocation_mode == 'highest':
            winner = highest_allocation(state=state, possible_agents=self.possible_agents,reserve_price=reserve_price)

        elif self.allocation_mode == 'deep':
            winner = self.allocation_net.compute_allocation(state=state, possible_agents=self.possible_agents,reserve_price=reserve_price)


        self.last_winner=winner


        return winner

    def compute_payment(self, state, winner=None):

        if self.action_mode == 'div':
            state = de_transform_state(state=state, args=self.args)

        if self.payment_mode == 'second_price':
            payment = second_price_rule(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'first_price':
            payment = first_price_rule(state, possible_agents=self.possible_agents, winner=winner)

        elif self.payment_mode == 'third_price':
            payment = third_price_rule(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'pay_by_submit':
            payment = pay_by_submit(state, possible_agents=self.possible_agents, winner=winner)
        elif self.payment_mode == 'deep':
            # add deep net payment
            payment = self.payment_net.compute_payment(state, possible_agents=self.possible_agents, winner=winner)

        else:
            print('not find the payment mode' + str(self.payment_mode))

        self.last_payment = payment

        return payment

    def assign_payment_net(self,payment_net):
        self.payment_net=payment_net

    def assign_allocation_net(self,allocation_net):
        self.allocation_net = allocation_net

    def export_payment_net(self):
        return deepcopy(self.payment_net)

    def export_allocation_net(self):
        return deepcopy(self.allocation_net)


    def close_grad(self):
        if self.payment_mode== 'deep' and self.payment_net is not None:
            self.payment_net.close_grad()

        if self.allocation_mode== 'deep' and self.allocation_net is not None:
            self.allocation_net.close_grad()


    def open_grad(self):

            if self.payment_mode == 'deep' and self.payment_net is not None:
                self.payment_net.open_grad()

            if self.allocation_mode == 'deep' and self.allocation_net is not None:
                self.allocation_net.open_grad()
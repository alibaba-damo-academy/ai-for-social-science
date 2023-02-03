import copy
import random
import numpy as np
# 写成类?
class budget_punish(object):
    def __init__(self, possible_agents=None,budget_sample=False,budget_sample_mode=None):
        self.possible_agents = possible_agents
        self.agent_budget_list = None
        self.agent_punish_list = None

        self.budget_sample=budget_sample
        self.budget_sample_mode=budget_sample_mode
        self.cur_budget={agent: None for agent in self.possible_agents}

        #
        if self.possible_agents is not None:
            self.agent_budget_list = {agent: None for agent in self.possible_agents}
            self.agent_punish_list = {agent: 0 for agent in self.possible_agents}

    def load_budget_profile(self, budget_profile):
        self.agent_budget_list = copy.deepcopy(budget_profile)

    def get_budget_profile(self):
        return self.agent_budget_list


    def load_budget_punish(self, punish_profile):
        self.agent_punish_list = copy.deepcopy(punish_profile)


    def generate_budeget(self,user_id):
        if (self.budget_sample == False) or (self.budget_sample_mode is None) or (user_id not in self.cur_budget):
            return
        elif self.budget_sample_mode == 'uniform':
            highest_budget = self.agent_budget_list[user_id]
            budget_realization = np.random.uniform(0, highest_budget)  # [0-H)
        else:
            print('not found sampled mode in budget module')
            budget_realization = 0

        self.cur_budget[user_id] = budget_realization

        return


    def get_budget_range(self,user_id):
        return self.agent_budget_list[user_id]

    def get_budget(self,user_id):
        # should add the current budget for agent optimization --->to be done



        if (self.budget_sample ==False ) or (self.budget_sample_mode is None):

            return     self.agent_budget_list[user_id]
        elif  (user_id not in self.cur_budget):
            print('not find ' + str(user_id) + 'in budget list')
            return None
        else:

            return self.cur_budget[user_id]


    def check_budget_result(self, user_id=None, allocation=0, payment=0):

        if (user_id is None) or (allocation == 0) or (self.agent_budget_list is None) or \
            (user_id not in self.agent_budget_list) or (self.agent_budget_list[user_id] is None) :

            # budget = None  equals no budget
            #print(1)
            return allocation, payment
        else:
            # check allocation

            detailed_budget = self.get_budget(user_id=user_id)

            if payment > detailed_budget:
                #exceed the payment, allocation =0 #not assign

                if (user_id not in self.agent_punish_list) or (self.agent_punish_list[user_id] is None) :
                   # print(2)
                    return 0,0
                else:
                    #print(3)
                    return 0,self.agent_punish_list[user_id]
            else:
                # not exceed the payment
                return allocation, payment


def load_agent_budget(args, agent_name_list):
    # can be extend to load from the file
    budget = {agent: None for agent in agent_name_list}
    if args.budget_param is not None and len(args.budget_param) > 0:
        last_budget = None
        last_agent = None
        for i in range(len(args.budget_param)):
            agent_budget = args.budget_param[i]
            agt_name = agent_name_list[i]
            budget[agt_name] = agent_budget

            last_budget = agent_budget
            last_agent = agt_name
        # set the rest of the budget into the last budget
        if last_budget is not None and last_agent != agent_name_list[-1]:
            for agt in agent_name_list:
                if budget[agt] is None:
                    # not assign budget
                    budget[agt] = last_budget
                    #print(agt)
    return budget


def load_agent_budget_punishment(args, agent_name_list):
    # can be extend to load from the file
    punish = {agent: None for agent in agent_name_list}
    if args.budget_param is not None and len(args.budget_punish_param) > 0:
        last_punish = None
        last_agent = None
        for i in range(len(args.budget_punish_param)):
            agent_punish = args.budget_punish_param[i]
            agt_name = agent_name_list[i]
            punish[agt_name] = agent_punish

            last_punish = agent_punish
            last_agent = agt_name
        # set the rest of the budget into the last budget_punish_param
        if last_punish is not None and last_agent != agent_name_list[-1]:
            for agt in agent_name_list:
                if punish[agt] is None:
                    # not assign budget
                    punish[agt] = last_punish
    return punish


def build_budget(args, agent_name_list=[]):

    budget_sample_list=[None,'uniform']
    budget_flag_list=[False,True]

    if args.budget_mode == 'budget_with_punish':

        budget = budget_punish(possible_agents=agent_name_list,\
                               budget_sample=budget_flag_list[args.budget_sampled_mode],\
                               budget_sample_mode=budget_sample_list[args.budget_sampled_mode]
                               )

        budget_profile = load_agent_budget(args, agent_name_list)
        budget_punishment = load_agent_budget_punishment(args, agent_name_list)

        budget.load_budget_profile(budget_profile=budget_profile)
        budget.load_budget_punish(punish_profile=budget_punishment)

        #init budget
        for agt in agent_name_list:
            budget.generate_budeget(user_id=agt)


    return budget



import math
import numpy as np


def check_support_reward_shaping(support_list, reward_function_name):
    if reward_function_name is None:
        return False
    else:
        return reward_function_name in support_list


def build_reward_function_config(args):
    if args.reward_shaping is not None:
        config = {}
        config['function'] = args.reward_shaping

        # build your parameter here

        # example a =0.5 :
        if args.reward_shaping_param is not None and len(args.reward_shaping_param) > 0:
            # print(args.reward_shaping_param)

            config['a'] = args.reward_shaping_param[0]

        return config

    return {}


def compute_reward(true_value, pay, allocation=0, pay_sign=-1,

                   reward_shaping=False, reward_function_config={}, user_id=None, budget=None, args=None, info=None
                   ):
    if 'other_value' in info:
        other_value = info['other_value']
    else:
        other_value =0



    mode = 'individual'
    if args.inner_cooperate == 1 and other_value is not None:  # can witness other bidder value:

        if args.value_div == 1 and info['cooperate_win']: # cooperate win | other wise allocation =0
            mode ='shared_value_with_pay'
            pay=info['cooperate_pay']

        if args.cooperate_pay_limit == 1 and allocation==1 and pay_sign * pay > true_value:  # win and exceed the pay limit
                allocation=0
                pay=0.1*pay # 10% punish
        true_value += other_value  # only the winner can added on, the rest allocation is 0 thus no influence

    # pay sign ==-1 ---> pay is negative

    reward = None
    payment = pay_sign * pay

    # budget reshape ()
    if budget is not None:
        # in punishment mode:  pay over bid will not receive the item and receive a punishment
        # print('---')
        # print(user_id)
        # print(allocation)
        # print(payment)
        allocation, payment = budget.check_budget_result(user_id=user_id, allocation=allocation, payment=payment)
    #

    # reward computing function
    if mode == 'individual':
        
        # Support both scalar (single item) and vector (multi-item) allocation
        reward = np.dot(true_value, allocation) - payment  # pay_sign * pay 

    elif mode == 'shared_value_with_pay': #already win

        reward = (true_value  - payment)*1.0 / len(args.inner_cooperate_id)   # shared value - shared payment





    # print('reward is ' +str(reward))
    # print('payment is ' + str(payment))
    # print('true_value is ' + str(true_value))
    # print('-----')

    ####### utility compute ####
    if reward_shaping:
        # print(reward_function_config)
        # print(reward)
        if 'function' in reward_function_config:
            if reward_function_config['function'] == 'CRRA':
                reward = CRRA_shaping(reward, a=reward_function_config['a'])
            elif reward_function_config['function'] == 'CARA':
                reward = CARA_shaping(reward, a=reward_function_config['a'])
        else:
            print('not find [function] in config, compute without reward shaping')

    return reward


def CRRA_shaping(reward, a=0.5):
    # u(z)=z^a,a in [0,1]
    # print(reward)
    if reward == 0 or reward is None:
        return 0
    else:
        if reward < 0:
            tmp = ((-1 * reward) ** a) * -1  # deal with nan
        else:
            tmp = reward ** a
        if tmp is np.nan:
            print('exist nan !')

        return tmp


def CARA_shaping(reward, a=0.5):
    # u(z)=1-exp^(-az),a>0
    return 1 - math.exp(-1 * a * reward)




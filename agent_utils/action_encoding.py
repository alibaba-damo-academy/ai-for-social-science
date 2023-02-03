from copy import deepcopy
def action_transform(action=0,mode='div',true_value=0,args=None):
        if mode =='div':
            if args is None or args.div_const is None:
                print('not args or division const for action transform')
                return action
            else:
                return true_value * args.bidding_range + action
        else:
            return action


def action_de_transform(transformed_action,args):
    # bid = value * (action) / div_const
    # e.g action in [0-200] , div const is 100, means 0%-200%
    true_value = int(transformed_action / args.bidding_range)
    action = transformed_action % args.bidding_range
    bidding_price = true_value * action *1.0 / args.div_const
    return bidding_price


def de_transform_state(state,args):
    new_state = deepcopy(state)
    for agt in state:
        true_bid = action_de_transform(transformed_action=state[agt], args=args)
        new_state[agt]=true_bid
    return new_state
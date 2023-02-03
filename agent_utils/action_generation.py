from .action_encoding import *


def action_generation(args, agt, obs):
    new_action = agt.generate_action(obs)  # get the next round action based on the observed budget
    if args.action_mode == 'div':
        new_action = action_transform(action=new_action, mode='div', true_value=agt.get_latest_true_value(), args=args)

    return new_action


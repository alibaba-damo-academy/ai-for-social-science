#python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 1 --folder_name 'CRRA' --reward_shaping 'CRRA' --env_iters 2000000 --reward_shaping_param 0.5 --reward_shaping_param 0.6
#python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 2 --folder_name 'CRRA' --reward_shaping 'CRRA' --env_iters 2000000 --reward_shaping_param 0.5 --reward_shaping_param 0.6


python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 1 --folder_name 'CARA' --reward_shaping 'CARA' --env_iters 2000010 --reward_shaping_param 0.5 --reward_shaping_param 0.6
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 2 --folder_name 'CARA' --reward_shaping 'CARA' --env_iters 2000010 --reward_shaping_param 0.5 --reward_shaping_param 0.6



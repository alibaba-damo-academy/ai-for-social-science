
# 3 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 0  --folder_name 'v2s/spectrum' \
--bidding_range 20 --valuation_range 10 --env_iters 1500000 \
--estimate_frequent 100000 --revenue_averaged_stamp 20000 --exploration_epoch 50000 \
--public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 --public_signal_lower_bound 0 --public_signal_spectrum 1 


# 3 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 1  --folder_name 'v2s/spectrum' \
--bidding_range 20 --valuation_range 10 --env_iters 1500000 \
--estimate_frequent 100000 --revenue_averaged_stamp 20000 --exploration_epoch 50000 \
--public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 --public_signal_lower_bound 3 --public_signal_spectrum 1 

# 3 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 2  --folder_name 'v2s/spectrum' \
--bidding_range 20 --valuation_range 10 --env_iters 1500000 \
--estimate_frequent 100000 --revenue_averaged_stamp 20000 --exploration_epoch 50000 \
--public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 --public_signal_lower_bound 6 --public_signal_spectrum 1 

# 3 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 3  --folder_name 'v2s/spectrum' \
--bidding_range 20 --valuation_range 10 --env_iters 1500000 \
--estimate_frequent 100000 --revenue_averaged_stamp 20000 --exploration_epoch 50000 \
--public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 --public_signal_lower_bound 9 --public_signal_spectrum 1 

# 3 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 4  --folder_name 'v2s/spectrum' \
--bidding_range 20 --valuation_range 10 --env_iters 1500000 \
--estimate_frequent 100000 --revenue_averaged_stamp 20000 --exploration_epoch 50000 \
--public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 --public_signal_lower_bound 10 --public_signal_spectrum 1 

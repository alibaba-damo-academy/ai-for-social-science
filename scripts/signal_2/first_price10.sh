# 2 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 121  --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 30000000 \
--estimate_frequent 2000000 --revenue_averaged_stamp 20000 --exploration_epoch 1000000 \
--public_signal_dim 2 --player_num 2 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 &

# 3 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 131  --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 30000000 \
--estimate_frequent 2000000 --revenue_averaged_stamp 20000 --exploration_epoch 1000000 \
--public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 &

# 4 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 141  --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 30000000 \
--estimate_frequent 2000000 --revenue_averaged_stamp 20000 --exploration_epoch 1000000 \
--public_signal_dim 4 --player_num 4 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 &

# 6 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 161  --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 150000000 \
--estimate_frequent 2000000 --revenue_averaged_stamp 20000 --exploration_epoch 5000000 \
--public_signal_dim 6 --player_num 6 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 &

# 10 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 1101  --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 1500000000 \
--estimate_frequent 20000000 --revenue_averaged_stamp 200000 --exploration_epoch 50000000 \
--public_signal_dim 10 --player_num 10 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20


# 2 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 122  --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 30000000 \
--estimate_frequent 2000000 --revenue_averaged_stamp 20000 --exploration_epoch 1000000 \
--public_signal_dim 4 --player_num 2 --agt_obs_public_signal_dim 2 \
--value_to_signal 1 --public_signal_range 20 &
# 3 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 132  --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 30000000 \
--estimate_frequent 2000000 --revenue_averaged_stamp 20000 --exploration_epoch 1000000 \
--public_signal_dim 6 --player_num 3 --agt_obs_public_signal_dim 2 \
--value_to_signal 1 --public_signal_range 20 & 
# 4 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 142  --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 30000000 \
--estimate_frequent 2000000 --revenue_averaged_stamp 20000 --exploration_epoch 1000000 \
--public_signal_dim 8 --player_num 4 --agt_obs_public_signal_dim 2 \
--value_to_signal 1 --public_signal_range 20 &
# 6 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 162  --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 150000000 \
--estimate_frequent 2000000 --revenue_averaged_stamp 20000 --exploration_epoch 5000000 \
--public_signal_dim 12 --player_num 6 --agt_obs_public_signal_dim 2 \
--value_to_signal 1 --public_signal_range 20 &

# 10 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 1102 --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 1500000000 \
--estimate_frequent 20000000 --revenue_averaged_stamp 200000 --exploration_epoch 50000000 \
--public_signal_dim 10 --player_num 10 --agt_obs_public_signal_dim 2 \
--value_to_signal 1 --public_signal_range 20
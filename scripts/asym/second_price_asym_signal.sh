# 3 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 320  --folder_name 'asym_signal' \
--bidding_range 20 --valuation_range 10 --env_iters 1500000 \
--estimate_frequent 100000 --revenue_averaged_stamp 20000 --exploration_epoch 50000 \
--public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 --public_signal_asym 1


# 3 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 340  --folder_name 'asym_signal' \
--bidding_range 40 --valuation_range 20 --env_iters 1500000 \
--estimate_frequent 100000 --revenue_averaged_stamp 20000 --exploration_epoch 50000 \
--public_signal 1 --public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 40 --public_signal_asym 1


# 5 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 540  --folder_name 'asym_signal' \
--bidding_range 40 --valuation_range 20 --env_iters 15000000 \
--estimate_frequent 1000000 --revenue_averaged_stamp 20000 --exploration_epoch 500000 \
--public_signal 1 --public_signal_dim 5 --player_num 5 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 40 --public_signal_asym 1


# 5 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 580  --folder_name 'asym_signal' \
--bidding_range 80 --valuation_range 40 --env_iters 15000000 \
--estimate_frequent 1000000 --revenue_averaged_stamp 20000 --exploration_epoch 500000 \
--public_signal 1 --public_signal_dim 5 --player_num 5 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 80 --public_signal_asym 1

# 5 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 5160  --folder_name 'asym_signal' \
--bidding_range 160 --valuation_range 80 --env_iters 15000000 \
--estimate_frequent 1000000 --revenue_averaged_stamp 20000 --exploration_epoch 500000 \
--public_signal 1 --public_signal_dim 5 --player_num 5 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 160 --public_signal_asym 1

# 10 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 1040  --folder_name 'asym_signal' \
--bidding_range 40 --valuation_range 20 --env_iters 15000000 \
--estimate_frequent 1000000 --revenue_averaged_stamp 20000 --exploration_epoch 500000 \
--public_signal 1 --public_signal_dim 10 --player_num 10 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 40 --public_signal_asym 1



############################

# 3 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 321  --folder_name 'asym_signal' \
--bidding_range 20 --valuation_range 10 --env_iters 1500000 \
--estimate_frequent 100000 --revenue_averaged_stamp 20000 --exploration_epoch 50000 \
--public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20 --public_signal_asym 1


# 3 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 341  --folder_name 'asym_signal' \
--bidding_range 40 --valuation_range 20 --env_iters 1500000 \
--estimate_frequent 100000 --revenue_averaged_stamp 20000 --exploration_epoch 50000 \
--public_signal 1 --public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 40 --public_signal_asym 1


# 5 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 541  --folder_name 'asym_signal' \
--bidding_range 40 --valuation_range 20 --env_iters 15000000 \
--estimate_frequent 1000000 --revenue_averaged_stamp 20000 --exploration_epoch 500000 \
--public_signal 1 --public_signal_dim 5 --player_num 5 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 40 --public_signal_asym 1


# 5 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 581  --folder_name 'asym_signal' \
--bidding_range 80 --valuation_range 40 --env_iters 15000000 \
--estimate_frequent 1000000 --revenue_averaged_stamp 20000 --exploration_epoch 500000 \
--public_signal 1 --public_signal_dim 5 --player_num 5 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 80 --public_signal_asym 1

# 5 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 5161  --folder_name 'asym_signal' \
--bidding_range 160 --valuation_range 80 --env_iters 15000000 \
--estimate_frequent 1000000 --revenue_averaged_stamp 20000 --exploration_epoch 500000 \
--public_signal 1 --public_signal_dim 5 --player_num 5 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 160 --public_signal_asym 1


# 10 players
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 1041  --folder_name 'asym_signal' \
--bidding_range 40 --valuation_range 20 --env_iters 15000000 \
--estimate_frequent 1000000 --revenue_averaged_stamp 20000 --exploration_epoch 500000 \
--public_signal 1 --public_signal_dim 10 --player_num 10 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 40 --public_signal_asym 1

######################

# different mean

# 3 players
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 340  --folder_name 'asym_signal' \
--bidding_range 40 --valuation_range 20 --env_iters 1500000 \
--estimate_frequent 100000 --revenue_averaged_stamp 20000 --exploration_epoch 50000 \
--public_signal 1 --public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 40 --public_signal_asym 1

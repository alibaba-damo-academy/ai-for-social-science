# First price auction with partial signal
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 1 \
--folder_name 'partial_signal/first_price' \
--bidding_range 20 --valuation_range 10 --env_iters 1000000  \
--record_efficiency 0 --estimate_frequent 200000 --revenue_averaged_stamp 20000 \
--exploration_epoch 100000 --player_num 4 --public_signal 1 \
--public_signal_dim 1 --agt_obs_public_signal_dim 1 

python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 2 \
--folder_name 'partial_signal/first_price' \
--bidding_range 20 --valuation_range 10 --env_iters 1000000  \
--record_efficiency 0 --estimate_frequent 200000 --revenue_averaged_stamp 20000 \
--exploration_epoch 100000 --player_num 4 --public_signal 1 \
--public_signal_dim 2 --agt_obs_public_signal_dim 2 

python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 3 \
--folder_name 'partial_signal/first_price' \
--bidding_range 20 --valuation_range 10 --env_iters 1000000  \
--record_efficiency 0 --estimate_frequent 200000 --revenue_averaged_stamp 20000 \
--exploration_epoch 100000 --player_num 4 --public_signal 1 \
--public_signal_dim 3 --agt_obs_public_signal_dim 3 

python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 4 \
--folder_name 'partial_signal/first_price' \
--bidding_range 20 --valuation_range 10 --env_iters 1000000  \
--record_efficiency 0 --estimate_frequent 200000 --revenue_averaged_stamp 20000 \
--exploration_epoch 100000 --player_num 4 --public_signal 1 \
--public_signal_dim 4 --agt_obs_public_signal_dim 4

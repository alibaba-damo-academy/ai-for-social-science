# First price auction with partial signal
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 1 \
--folder_name 'partial_signal/first_price' \
--bidding_range 20 --valuation_range 10 --env_iters 40000000  \
--record_efficiency 0 --estimate_frequent 3000000 --revenue_averaged_stamp 40000 \
--exploration_epoch 20000 --player_num 2 --public_signal 1 --public_signal_dim --public_signal_range 5 


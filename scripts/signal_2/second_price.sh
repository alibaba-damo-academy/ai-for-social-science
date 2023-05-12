cd ..
cd ..
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 2  --folder_name 'v2s' \
--bidding_range 20 --valuation_range 10 --env_iters 3000000 \
--estimate_frequent 200000 --revenue_averaged_stamp 20000 --exploration_epoch 100000 \
--public_signal_dim 3 --player_num 3 --agt_obs_public_signal_dim 1 \
--value_to_signal 1 --public_signal_range 20
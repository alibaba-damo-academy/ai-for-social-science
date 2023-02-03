cd ..
cd ..
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 222  --folder_name 'signal' \
--bidding_range 20 --valuation_range 10 --env_iters 3000000 \
--estimate_frequent 200000 --revenue_averaged_stamp 20000 --exploration_epoch 100000 --player_num 4 \
--public_signal_dim 3 --agt_obs_public_signal_dim 3
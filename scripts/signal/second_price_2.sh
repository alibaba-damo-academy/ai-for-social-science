cd ..
cd ..
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 21  --folder_name 'signal' \
--bidding_range 20 --valuation_range 10 --env_iters 12000000 \
--estimate_frequent 800000 --revenue_averaged_stamp 20000 --exploration_epoch 400000 --player_num 3 \
--public_signal_dim 2 --agt_obs_public_signal_dim 2
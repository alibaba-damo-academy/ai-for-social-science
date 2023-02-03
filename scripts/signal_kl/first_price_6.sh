cd ..
cd ..
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 6  --folder_name 'kl_signal' \
--bidding_range 20 --valuation_range 10 --env_iters 20000000 \
--estimate_frequent 2000000 --revenue_averaged_stamp 20000 --exploration_epoch 1000000 \
--public_signal_dim 6 --player_num 6 --agt_obs_public_signal_dim 1 \
--value_generator_mode 'mean' --public_signal_range 10
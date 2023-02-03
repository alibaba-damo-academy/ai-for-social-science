cd ..
cd ..
cd ..
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 1  --folder_name 'kl_sp_signal' \
--bidding_range 20 --valuation_range 10 --env_iters 3000000 \
--estimate_frequent 200000 --revenue_averaged_stamp 20000 --exploration_epoch 100000 --player_num 2 \
--public_signal_dim 2  --agt_obs_public_signal_dim 1 \
--value_generator_mode 'mean' --public_signal_range 10 \
--speicial_agt 'player_0'
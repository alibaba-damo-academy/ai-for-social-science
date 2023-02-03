cd ..
cd ..
cd ..
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 3  --folder_name 'kl_sp_signal' \
--bidding_range 20 --valuation_range 10 --env_iters 10000000 \
--estimate_frequent 1000000 --revenue_averaged_stamp 20000 --exploration_epoch 200000 --player_num 3 \
--public_signal_dim 3  --agt_obs_public_signal_dim 1 \
--value_generator_mode 'mean' --public_signal_range 10 \
--speicial_agt 'player_0'
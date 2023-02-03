cd ..
cd ..
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 3  --folder_name 'kl_single' \
--bidding_range 20 --valuation_range 10 --env_iters 3000000 \
--estimate_frequent 200000 --revenue_averaged_stamp 20000 --exploration_epoch 100000 --player_num 3 \
--public_signal_dim 3  --agt_obs_public_signal_dim 1 \
--value_generator_mode 'single' --public_signal_range 10



python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 4  --folder_name 'kl_single' \
--bidding_range 20 --valuation_range 10 --env_iters 3000000 \
--estimate_frequent 200000 --revenue_averaged_stamp 20000 --exploration_epoch 100000 --player_num 4 \
--public_signal_dim 4  --agt_obs_public_signal_dim 1 \
--value_generator_mode 'single' --public_signal_range 10



python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 44  --folder_name 'kl_single' \
--bidding_range 20 --valuation_range 10 --env_iters 20000000 \
--estimate_frequent 800000 --revenue_averaged_stamp 20000 --exploration_epoch 400000 --player_num 4 \
--public_signal_dim 4  --agt_obs_public_signal_dim 1 \
--value_generator_mode 'single' --public_signal_range 10


python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 33  --folder_name 'kl_single' \
--bidding_range 20 --valuation_range 10 --env_iters 20000000 \
--estimate_frequent 800000 --revenue_averaged_stamp 20000 --exploration_epoch 400000 --player_num 3 \
--public_signal_dim 3  --agt_obs_public_signal_dim 1 \
--value_generator_mode 'single' --public_signal_range 10
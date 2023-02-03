
cd ..
cd ..
python auction_bidding_simulate_multiple.py --mechanism 'second_price' --exp_id 5  --folder_name 'multi_round/private_value' \
--bidding_range 10 --valuation_range 10 --env_iters 1 --overbid True \
--round 2 \
--estimate_frequent 20000 --revenue_averaged_stamp 1000 --exploration_epoch 1 --player_num 5 \
--step_floor 10000 --item_num 2
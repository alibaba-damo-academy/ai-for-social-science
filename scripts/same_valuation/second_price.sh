cd ..
cd ..
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 2  --folder_name 'exp' \
--bidding_range 10 --valuation_range 10 --env_iters 6000 \
--estimate_frequent 400 --revenue_averaged_stamp 20000 --exploration_epoch 200 --player_num 4
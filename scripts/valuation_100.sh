#python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 11  --bidding_range 200 --valuation_range 100 --env_iters 10000000 --estimate_frequent 400003 --revenue_averaged_stamp 2000
python auction_bidding_simulate.py --mechanism 'third_price' --exp_id 3  --folder_name 'exp' \
--bidding_range 10 --valuation_range 10 --env_iters 60000000 \
--estimate_frequent 800004 --revenue_averaged_stamp 20000 --exploration_epoch 1000000
python auction_bidding_simulate_multiple.py --allocation_mode 'vcg' --mechanism 'vcg' --exp_id 1  --folder_name 'nround_nitem' \
--bidding_range 10 --valuation_range 10 --env_iters 40000 --overbid True \
--round 1 \
--estimate_frequent 1000 --revenue_averaged_stamp 100 --exploration_epoch 4000 --player_num 5 \
--step_floor 10000 --item_num 5 --multi_item_decay 0.9

python auction_bidding_simulate_multiple.py --allocation_mode 'vcg' --mechanism 'vcg' --exp_id 11  --folder_name 'nround_nitem' \
--bidding_range 10 --valuation_range 10 --env_iters 40000 --overbid True \
--round 1 \
--estimate_frequent 1000 --revenue_averaged_stamp 100 --exploration_epoch 4000 --player_num 5 \
--step_floor 10000 --item_num 5 --multi_item_decay 0.7

python auction_bidding_simulate_multiple.py --mechanism 'first_price' --exp_id 2  --folder_name 'nround_nitem' \
--bidding_range 10 --valuation_range 10 --env_iters 40000 --overbid True \
--round 1 \
--estimate_frequent 1000 --revenue_averaged_stamp 100 --exploration_epoch 4000 --player_num 5 \
--step_floor 10000 --item_num 5 --multi_item_decay 0.9

python auction_bidding_simulate_multiple.py --mechanism 'first_price' --exp_id 21  --folder_name 'nround_nitem' \
--bidding_range 10 --valuation_range 10 --env_iters 40000 --overbid True \
--round 1 \
--estimate_frequent 1000 --revenue_averaged_stamp 100 --exploration_epoch 4000 --player_num 5 \
--step_floor 10000 --item_num 5 --multi_item_decay 0.7

python auction_bidding_simulate_multiple.py --mechanism 'second_price' --exp_id 3  --folder_name 'nround_nitem' \
--bidding_range 10 --valuation_range 10 --env_iters 40000 --overbid True \
--round 1 \
--estimate_frequent 1000 --revenue_averaged_stamp 100 --exploration_epoch 4000 --player_num 5 \
--step_floor 10000 --item_num 5 --multi_item_decay 0.9

python auction_bidding_simulate_multiple.py --mechanism 'second_price' --exp_id 31  --folder_name 'nround_nitem' \
--bidding_range 10 --valuation_range 10 --env_iters 40000 --overbid True \
--round 1 \
--estimate_frequent 1000 --revenue_averaged_stamp 100 --exploration_epoch 4000 --player_num 5 \
--step_floor 10000 --item_num 5 --multi_item_decay 0.7







python auction_bidding_simulate_multiple.py --allocation_mode 'vcg' --mechanism 'vcg' --exp_id 2  --folder_name 'test_yang' \
--bidding_range 10 --valuation_range 10 --env_iters 4000 --overbid True \
--round 1 \
--estimate_frequent 100 --revenue_averaged_stamp 10 --exploration_epoch 400 --player_num 5 \
--step_floor 1000 --item_num 5 --multi_item_decay 0.9




python auction_bidding_simulate_multiple.py --allocation_mode 'vcg' --mechanism 'vcg' --exp_id 1  --folder_name 'nround_nitem' \
--bidding_range 10 --valuation_range 10 --env_iters 1000 --overbid True \
--round 1 \
--estimate_frequent 20 --revenue_averaged_stamp 10 --exploration_epoch 10 --player_num 5 \
--step_floor 10 --item_num 5 --multi_item_decay 0.9



python auction_bidding_simulate_multiple.py --allocation_mode 'vcg' --mechanism 'vcg' --exp_id 1  --folder_name 'nround_nitem' \
--bidding_range 10 --valuation_range 10 --env_iters 1 --overbid True \
--round 1 \
--estimate_frequent 20 --revenue_averaged_stamp 10 --exploration_epoch 1 --player_num 5 \
--step_floor 10 --item_num 5 --multi_item_decay 0.9


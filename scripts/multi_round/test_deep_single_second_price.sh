cd ..
cd ..
python auction_bidding_simulate_multiple.py --mechanism 'second_price' --exp_id 3200  --folder_name 'deep_test_multi_env' \
--bidding_range 10 --valuation_range 10 --env_iters 1000000 --overbid True \
--round 1 \
--estimate_frequent 20000 --revenue_averaged_stamp 1000 --exploration_epoch 1 --player_num 5 \
--step_floor 10000 \
--algorithm 'deep' --lr 3e-3 --update_frequent 256 --multi_item_decay 0.5

#
#python auction_bidding_simulate_multiple.py --mechanism 'second_price' --exp_id 33  --folder_name 'deep_test_multi_env' \
#--bidding_range 10 --valuation_range 10 --env_iters 1000000 --overbid True \
#--round 1 \
#--estimate_frequent 20000 --revenue_averaged_stamp 1000 --exploration_epoch 100000 --player_num 5 \
#--step_floor 10000 \
##--algorithm 'deep' --lr 1e-2 --update_frequent 2000


#50 30 20
#20 50 30
#

#--budget_param 10.0 --budget_param 20.0 --budget_param 50.0 --budget_param 100.0 --budget_param 200.0 \

#python auction_bidding_simulate_multiple.py --mechanism 'second_price' --exp_id 5  --folder_name 'Multi_round_exp' \
#--bidding_range 10 --valuation_range 10 --env_iters 1 --overbid True \
#--round 100000 \
#--estimate_frequent 20000 --revenue_averaged_stamp 1000 --exploration_epoch 1 --player_num 5 \
#--step_floor 10000 \
#--budget_mode 'budget_with_punish'  \
#--budget_param 2000.0 --budget_param 2000.0 --budget_param 2000.0 --budget_param 2000.0 --budget_param 2000.0 \
#--budget_punish_param 0.01


#python auction_bidding_simulate_multiple.py --mechanism 'second_price' --exp_id 101  --folder_name 'Multi_round_exp' \
#--bidding_range 10 --valuation_range 10 --env_iters 1 --overbid True \
#--round 100000 \
#--estimate_frequent 20000 --revenue_averaged_stamp 1000 --exploration_epoch 1 --player_num 5 \
#--step_floor 10000 --smart_step_floor 'smart' \
#--budget_mode 'budget_with_punish'  \
#--budget_param 10.0 --budget_param 20.0 --budget_param 50.0 --budget_param 100.0 --budget_param 200.0 \
#--budget_punish_param 0.01 --extra_round_income 0.001
cd ..
cd ..
#python auction_bidding_simulate_multiple.py --mechanism 'second_price' --exp_id 5  --folder_name 'Multi_round_exp_selfplay' \
#--bidding_range 10 --valuation_range 10 --env_iters 1 --overbid True \
#--round 100000 \
#--estimate_frequent 20000 --revenue_averaged_stamp 1000 --exploration_epoch 1 --player_num 5 \
#--step_floor 10000 \
#--budget_mode 'budget_with_punish'  \
#--budget_param 2000.0 --budget_param 2000.0 --budget_param 2000.0 --budget_param 2000.0 --budget_param 2000.0 \
#--budget_punish_param 0.01 \
#--self_play 1 --self_play_id 0

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


python auction_bidding_simulate_multiple.py --mechanism 'second_price' --exp_id 101  --folder_name 'Multi_round_exp_selfplay' \
--bidding_range 10 --valuation_range 10 --env_iters 1 --overbid True \
--round 100000 \
--estimate_frequent 20000 --revenue_averaged_stamp 1000 --exploration_epoch 1 --player_num 5 \
--step_floor 10000 --smart_step_floor 'smart' \
--budget_mode 'budget_with_punish'  \
--budget_param 10.0 --budget_param 20.0 --budget_param 50.0 --budget_param 100.0 --budget_param 200.0 \
--budget_punish_param 0.01 --extra_round_income 0.001 \
--self_play 1 --self_play_id 0 --self_play_id 1
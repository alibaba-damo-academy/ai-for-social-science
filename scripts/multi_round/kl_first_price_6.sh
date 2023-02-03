cd ..
cd ..

# exp1
#python auction_bidding_simulate_multiple.py --mechanism 'first_price' --exp_id 5  --folder_name 'Multi_round_exp_kl' \
#--bidding_range 30 --valuation_range 10 --env_iters 1 --overbid True \
#--round 200000 \
#--estimate_frequent 20000 --revenue_averaged_stamp 1000 --exploration_epoch 1 --player_num 5 \
#--step_floor 10000 \
#--budget_mode 'budget_with_punish'  \
#--budget_param 2000.0 --budget_param 2000.0 --budget_param 2000.0 --budget_param 2000.0 --budget_param 2000.0 \
#--budget_punish_param 0.001 \
#--public_signal 1 --public_signal_dim 5 --player_num 5 --agt_obs_public_signal_dim 1 \
#--value_generator_mode 'mean' --public_signal_range 10 \
#--smart_step_floor 'smart' --extra_round_income 0.001

#
## exp2
#python auction_bidding_simulate_multiple.py --mechanism 'first_price' --exp_id 52  --folder_name 'Multi_round_exp_kl' \
#--bidding_range 30 --valuation_range 10 --env_iters 1 --overbid True \
#--round 200000 \
#--estimate_frequent 20000 --revenue_averaged_stamp 1000 --exploration_epoch 1 --player_num 5 \
#--step_floor 10000 \
#--budget_mode 'budget_with_punish'  \
#--budget_param 10.0 --budget_param 50.0 --budget_param 100.0 --budget_param 500.0 --budget_param 2000.0 \
#--budget_punish_param 0.001 \
#--public_signal 1 --public_signal_dim 5 --player_num 5 --agt_obs_public_signal_dim 1 \
#--value_generator_mode 'mean' --public_signal_range 10 \
#--smart_step_floor 'smart' --extra_round_income 0.001

#exp3 --value_generator_mode 'single_u'
#
python auction_bidding_simulate_multiple.py --mechanism 'first_price' --exp_id 53  --folder_name 'Multi_round_exp_kl_sp_signal' \
--bidding_range 20 --valuation_range 10 --env_iters 1 --overbid True \
--round 400000 \
--estimate_frequent 20000 --revenue_averaged_stamp 1000 --exploration_epoch 1 --player_num 5 \
--step_floor 100000 \
--budget_mode 'budget_with_punish'  \
--budget_param 20000.0 --budget_param 20000.0 --budget_param 20000.0 --budget_param 20000.0 --budget_param 20000.0 \
--budget_punish_param 0.001 \
--public_signal 1 --public_signal_dim 5 --player_num 5 --agt_obs_public_signal_dim 1 \
--public_signal_range 10 \
--smart_step_floor 'smart' \
--extra_round_income 0.01 \
--value_generator_mode 'mean' \
--speicial_agt 'player_0'
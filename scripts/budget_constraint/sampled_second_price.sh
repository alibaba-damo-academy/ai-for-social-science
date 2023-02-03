cd ..
cd ..
## require more  time to converge due to the huge space and multiple solver
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 2  --folder_name 'budget_sample' \
--bidding_range 10 --valuation_range 10 --env_iters 30000000 \
--estimate_frequent 1000000 --revenue_averaged_stamp 200000 --exploration_epoch 100000 --player_num 2 \
--budget_mode 'budget_with_punish'  \
--budget_param 10.0  \
--budget_punish_param 1.0 --budget_sampled_mode 1


#
##
#python auction_bidding_simulate.py --mechanism second_price --exp_id 2  --folder_name budget \
#--bidding_range 10 --valuation_range 10 --env_iters 4000000 \
#--estimate_frequent 400000 --revenue_averaged_stamp 20000 --exploration_epoch 100000 --player_num 4 \
#--budget_mode budget_with_punish  \
#--budget_param 5.0 --budget_param 5.0 \
#--budget_punish_param 1.0
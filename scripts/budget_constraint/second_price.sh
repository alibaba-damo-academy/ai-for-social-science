cd ..
cd ..

python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 222  --folder_name 'budget' \
--bidding_range 100 --valuation_range 100 --env_iters 8000000 \
--estimate_frequent 400000 --revenue_averaged_stamp 20000 --exploration_epoch 100000 --player_num 4 \
--budget_mode 'budget_with_punish'  \
--budget_param 50.0 --budget_param 50.0 \
--budget_punish_param 1.0



#
##
#python auction_bidding_simulate.py --mechanism second_price --exp_id 2  --folder_name budget \
#--bidding_range 10 --valuation_range 10 --env_iters 4000000 \
#--estimate_frequent 400000 --revenue_averaged_stamp 20000 --exploration_epoch 100000 --player_num 4 \
#--budget_mode budget_with_punish  \
#--budget_param 5.0 --budget_param 5.0 \
#--budget_punish_param 1.0
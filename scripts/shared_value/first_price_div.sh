cd ..
cd ..

#python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 22  --folder_name 'budget' \
#--bidding_range 100 --valuation_range 100 --env_iters 8000000 \
#--estimate_frequent 400000 --revenue_averaged_stamp 20000 --exploration_epoch 100000 --player_num 4 \
#--budget_mode 'budget_with_punish'  \
#--budget_param 50.0 --budget_param 50.0 \
#--budget_punish_param 1.0




#
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 100  --folder_name 'shared_value' \
--bidding_range 60 --valuation_range 10 --env_iters 16000000  --record_efficiency 1 \
--estimate_frequent 800000 --revenue_averaged_stamp 40000 --exploration_epoch 200000 --player_num 3 \
--assigned_valuation 60 --assigned_valuation 40 --assigned_valuation 20 \
--inner_cooperate 1 --overbid True --value_div 1 \
--inner_cooperate_id 1 --inner_cooperate_id 2

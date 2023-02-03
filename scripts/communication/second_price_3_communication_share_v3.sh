cd ..
cd ..

#python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 22  --folder_name 'budget' \
#--bidding_range 100 --valuation_range 100 --env_iters 8000000 \
#--estimate_frequent 400000 --revenue_averaged_stamp 20000 --exploration_epoch 100000 --player_num 4 \
#--budget_mode 'budget_with_punish'  \
#--budget_param 50.0 --budget_param 50.0 \
#--budget_punish_param 1.0




# no overbid
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 20  --folder_name 'commu_v3' \
--bidding_range 60 --valuation_range 10 --env_iters 30000000  --record_efficiency 1 \
--estimate_frequent 2000000 --revenue_averaged_stamp 40000 --exploration_epoch 20000 --player_num 3 \
--assigned_valuation 60 --assigned_valuation 40 --assigned_valuation 20 \
--communication 1 --communication_type 'value' \
--cm_id 1 --cm_id 2 \
--inner_cooperate 1 --overbid True --value_div 1 --cooperate_pay_limit 1 \
--inner_cooperate_id 1 --inner_cooperate_id 2

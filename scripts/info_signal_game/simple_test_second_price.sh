cd ..
cd ..


# normal signal
#python auction_bidding_simulate_multiple.py --mechanism 'first_price' --exp_id 0  --folder_name 'signal_game_env' \
#--bidding_range 20 --valuation_range 20 --env_iters 1000000 --overbid True \
#--round 1 \
#--estimate_frequent 200000 --revenue_averaged_stamp 1000 --exploration_epoch 100000 --player_num 4 \
#--step_floor 10000 \
#--env_name 'signal_game'


## assign valuation

python auction_bidding_simulate_multiple.py --mechanism 'first_price' --exp_id 1  --folder_name 'signal_game_env' \
--bidding_range 40 --valuation_range 20 --env_iters 3000000 --overbid True \
--assigned_valuation 10 --assigned_valuation 20 --assigned_valuation 30 --assigned_valuation 40 \
--round 1 \
--estimate_frequent 600000 --revenue_averaged_stamp 1000 --exploration_epoch 200000 --player_num 4 \
--step_floor 10000 \
--env_name 'signal_game'


# Fixed 
python auction_bidding_simulate_common_value.py --mechanism 'first_price' --value-mechanism 'common_fixed' --exp_id 1 --folder_name 'winner_first' 
python auction_bidding_simulate_common_value.py --mechanism 'second_price' --value-mechanism 'common_fixed' --exp_id 1 --folder_name 'winner_second' 

# Perturbed
python auction_bidding_simulate_common_value.py --mechanism 'first_price' --value-mechanism 'common_perturbed' --exp_id 1 --folder_name 'winner_first' 
python auction_bidding_simulate_common_value.py --mechanism 'second_price' --value-mechanism 'common_perturbed' --exp_id 1 --folder_name 'winner_second' 

# Baseline
python auction_bidding_simulate.py --mechanism 'first_price' --exp_id 1 --folder_name 'test_yang' 
python auction_bidding_simulate.py --mechanism 'second_price' --exp_id 2 --folder_name 'test_yang' 


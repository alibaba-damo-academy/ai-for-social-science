python examples/rllib_deep.py --algorithm 'DQN' --folder-name 'results/deep/'
python examples/rllib_deep.py --algorithm 'PPO' --folder-name 'results/deep/' 
python examples/rllib_deep.py --algorithm 'A2C' --folder-name 'results/deep/' 
python examples/rllib_deep.py --algorithm 'A3C' --folder-name 'results/deep/' --num_gpus 1 # num_works
python examples/rllib_deep.py --algorithm 'AlphaZero' --folder-name 'results/deep/' # action_masks
python examples/rllib_deep.py --algorithm 'APPO' --folder-name 'results/deep/' # MultGPULearner Thread 
python examples/rllib_deep.py --algorithm 'ARS' --folder-name 'results/deep/' --num_gpus 1  # Num row out workers
python examples/rllib_deep.py --algorithm 'BC' --folder-name 'results/deep/' --num_gpus 1 --stop-iters 20000
python examples/rllib_deep.py --algorithm 'TS' --folder-name 'results/deep/' # ??? 
python examples/rllib_deep.py --algorithm 'CRR' --folder-name 'results/deep/' # index 
python examples/rllib_deep.py --algorithm 'ES' --folder-name 'results/deep/' --num_gpus 1 # num_rollout_workers
python examples/rllib_deep.py --algorithm 'Rainbow' --folder-name 'results/deep/' # Unknown
python examples/rllib_deep.py --algorithm 'APEX-DQN' --folder-name 'results/deep/' # Unknown
python examples/rllib_deep.py --algorithm 'IMPALA' --folder-name 'results/deep/' # self.devices issue
python examples/rllib_deep.py --algorithm 'MARWIL' --folder-name 'results/deep/' --num_gpus 1 --stop-iters 20000
python examples/rllib_deep.py --algorithm 'PG' --folder-name 'results/deep/'
python examples/rllib_deep.py --algorithm 'SAC' --folder-name 'results/deep/'



python examples/rllib_deep.py --player-num 2 --algorithm 'DQN' --folder-name 'results/deep/'
python examples/rllib_deep.py --player-num 2 --algorithm 'PPO' --folder-name 'results/deep/' 
python examples/rllib_deep.py --player-num 2 --algorithm 'A2C' --folder-name 'results/deep/' 
python examples/rllib_deep.py --player-num 2 --algorithm 'BC' --folder-name 'results/deep/' --num_gpus 1 --stop-iters 20000
python examples/rllib_deep.py --player-num 2 --algorithm 'MARWIL' --folder-name 'results/deep/' --num_gpus 1 --stop-iters 20000
python examples/rllib_deep.py --player-num 2 --algorithm 'PG' --folder-name 'results/deep/'
python examples/rllib_deep.py --player-num 2 --algorithm 'SAC' --folder-name 'results/deep/'




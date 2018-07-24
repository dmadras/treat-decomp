xargs -n 16 -P 5 srun --gres=gpu:1 -p gpu < sweeps/18ju23_synth_prop_exp/args.txt

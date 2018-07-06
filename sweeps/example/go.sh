xargs -n 20 -P 10 srun --gres=gpu:1 -x guppy9,guppy10,guppy18,guppy25,guppy13,guppy22 -p gpuc < sweeps/icml_rebuttal_2out_repro/args_0.txt
xargs -n 19 -P 10 srun --gres=gpu:1 -x guppy9,guppy10,guppy18,guppy25,guppy13,guppy22 -p gpuc < sweeps/icml_rebuttal_2out_repro/args_1.txt

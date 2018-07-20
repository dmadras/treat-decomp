#!/bin/bash

synth_dsets="synth_basic synth_offset synth_confa_trmt synth_confa synth_confhid synth_corraz"

for syd in $synth_dsets;
do
    python src/run_make_synth_dataset.py -data $syd
    expname="$syd"_decomps
    python src/main.py -d $syd -n $expname
    python src/plot_decomposition.py -n $expname
done

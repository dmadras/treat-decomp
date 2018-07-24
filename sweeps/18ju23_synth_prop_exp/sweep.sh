#!/usr/bin/env bash

run_script_template=$1
dir_record_template=$2
expdir=$3 #/ais/gobi6/madras/learning-to-defer/experiments

dsets="compas health"
defer_pc=`seq -0.25 0.05 0.20`
defer_fc=`seq 0.0 0.07 0.49`
rej_pc="`seq 0 0.1 0.7` 0.45 0.55 0.65"
rej_fc=`seq 0.0 0.1 0.8`
numruns=5
runs=`seq 1 $numruns`
dmtypes="highacc lowacc highbias"

for dset in $dsets
do
for dmtype in $dmtypes
do
for pc in $defer_pc
do
for fc in $defer_fc
do
for run in $runs
do
fname=icml_rebuttal_2out_repro_sweep_v2_defer_nn_dset-"$dset"_dm-"$dmtype"_fc-"$fc"_pc-"$pc"_run-"$run"_mseed-"$run"_dseed-"$run"
fullexpdirname=$expdir/$fname
echo $fullexpdirname >> "$dir_record_template"_0.txt
if [ ! -e $fullexpdirname/test_results.csv ]; then
echo  "python src/run_model.py -d "$dset" -n $fname -pc $pc -fc $fc -pass -ua -to -def -mrs $run -drs $run" -dm "$dmtype">> "$run_script_template"_0.txt
fi
done
done
done
done

for dmtype in $dmtypes
do
for pc in $rej_pc
do
for fc in $rej_fc
do
for run in $runs
do

fname=icml_rebuttal_2out_repro_sweep_v2_reject_nn_dset-"$dset"_dm-"$dmtype"_fc-"$fc"_pc-"$pc"_run-"$run"_mseed-"$run"_dseed-"$run"
fullexpdirname=$expdir/$fname
echo $fullexpdirname >> "$dir_record_template"_1.txt
if [ ! -e $fullexpdirname/test_results.csv ]; then
echo "python src/run_model.py -d "$dset" -n $fname -pc $pc -fc $fc -pass -ua -to -mrs $run -drs $run" -dm "$dmtype" >> "$run_script_template"_1.txt
fi
done
done
done
done


done


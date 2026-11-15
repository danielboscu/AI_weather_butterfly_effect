#!/bin/bash
#PBS -N greedy_tf32
#PBS -A URIC0009
#PBS -j oe
#PBS -q main
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=64:mem=256GB:ngpus=4

module load conda
conda activate backup_pt212gpu_conda
module load cuda
unset NVIDIA_TF32_OVERRIDE

cd /glade/u/home/$USER/AI_weather_butterfly_effect/pangu
export pangu_scratch=/glade/derecho/scratch/$USER/sensitivity_analysis/pangu

for d in 0 1 2 3; do
    python3 inference/greedy/$script --device $d --shard $d --num_shards 4 --members $members --scale $scale --tf32 1 --out_root tf32 &
done
wait

#!/bin/bash
#PBS -N distill_greedy
#PBS -A URIC0009
#PBS -j oe
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=64GB

module load conda
conda activate backup_pt212gpu_conda

cd /glade/u/home/$USER/AI_weather_butterfly_effect/pangu
export pangu_scratch=/glade/derecho/scratch/$USER/sensitivity_analysis/pangu

python3 analysis/distill_run.py --precision $precision --mode greedy --source $source --noise $noise --scale $scale --members $members

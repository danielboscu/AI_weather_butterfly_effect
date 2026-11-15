#!/bin/bash
#PBS -N pangu_dke_compare
#PBS -A URIC0009
#PBS -j oe
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=128GB:ngpus=0

module load conda
conda activate backup_pt212gpu_conda
module load texlive/20250308

cd /glade/u/home/$USER/AI_weather_butterfly_effect/pangu
export pangu_scratch=/glade/derecho/scratch/$USER/sensitivity_analysis/pangu

python3 analysis/greedy_dke_compare_oper_g_oper_eda_0.001.py

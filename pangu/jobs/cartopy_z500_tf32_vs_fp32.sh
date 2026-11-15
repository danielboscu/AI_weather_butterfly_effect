#!/bin/bash
#PBS -N z500_tf32_cmp
#PBS -A URIC0009
#PBS -j oe
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=128GB:ngpus=0

module load conda
conda activate pt212gpu_conda
module load texlive/20250308

cd /glade/u/home/$USER/AI_weather_butterfly_effect/pangu
export pangu_scratch=/glade/derecho/scratch/$USER/sensitivity_analysis/pangu

python3 analysis/cartopy_z500_tf32_vs_fp32.py

#!/bin/bash
#PBS -N dke_tf32_fp32_pres
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

python3 analysis/dke_autoreg_eda_vs_g_tf32_vs_fp32_pres.py

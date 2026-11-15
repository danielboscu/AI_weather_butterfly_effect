#!/bin/bash
#SBATCH --job-name=to_npy
#SBATCH --account=TG-ATM170020
#SBATCH --partition=skx-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

cd $STOCKYARD/AI_weather_butterfly_effect/aifs
source .venv/bin/activate
export aifs_scratch=$SCRATCH/aifs

python3 analysis/grib_to_npy_members.py --precision $precision --noise-std $noise_std

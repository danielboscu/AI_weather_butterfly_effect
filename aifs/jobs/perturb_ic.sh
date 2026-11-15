#!/bin/bash
#SBATCH --job-name=perturb_ic
#SBATCH --account=TG-ATM170020
#SBATCH --partition=skx-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

cd $STOCKYARD/AI_weather_butterfly_effect/aifs
source .venv/bin/activate
export aifs_scratch=$SCRATCH/aifs

python3 data/perturb_ic.py --noise-std $noise_std

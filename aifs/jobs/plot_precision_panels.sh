#!/bin/bash
#SBATCH --job-name=prec_panels
#SBATCH --account=TG-ATM170020
#SBATCH --partition=skx-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

cd $STOCKYARD/AI_weather_butterfly_effect/aifs
source .venv/bin/activate
export aifs_scratch=$SCRATCH/aifs

python3 analysis/plot_precision_panels.py

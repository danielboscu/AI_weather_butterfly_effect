#!/bin/bash
#SBATCH --job-name=run_members
#SBATCH --account=TG-ATM170020
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --array=0-3

module load gcc/13.2.0
module load cuda/12.4

cd $STOCKYARD/AI_weather_butterfly_effect/aifs
source .venv-blackwell/bin/activate
export aifs_scratch=$SCRATCH/aifs
export ANEMOI_INFERENCE_NUM_CHUNKS=16

mkdir -p $aifs_scratch/members_pert
for member in $(seq $SLURM_ARRAY_TASK_ID 4 49); do
    python3 inference/run_member.py inference/member_$precision.yaml \
        input.grib=$aifs_scratch/ics_pert/ic_2021062600_pert_${noise_std}_${member}.grib \
        output.grib=$aifs_scratch/members_pert/member_${precision}_${noise_std}_${member}.grib
done

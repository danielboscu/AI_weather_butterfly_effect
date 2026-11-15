#!/bin/bash

precision=$1
cd $STOCKYARD/AI_weather_butterfly_effect/aifs/jobs

for noise_std in 1.0 0.1 0.01 0.001 0.0001; do
    for task in 0 1 2 3; do
        while [ $(squeue -u $USER -h -p rtx-small | wc -l) -ge 2 ]; do
            sleep 120
        done
        job=$(sbatch --parsable --array=$task-$task --export=ALL,precision=$precision,noise_std=$noise_std run_members.sh | tail -n1)
    done
    sbatch --dependency=afterok:$job --export=ALL,precision=$precision,noise_std=$noise_std to_npy.sh
done

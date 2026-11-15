import os
import argparse
import numpy as np
import onnxruntime as ort
import gc

pangu_scratch = os.environ["pangu_scratch"]
print(ort.get_available_providers())

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device_id this process binds to')
parser.add_argument('--shard', type=int, default=0, help='index of this worker among --num_shards')
parser.add_argument('--num_shards', type=int, default=1, help='total workers splitting the members across GPUs')
parser.add_argument('--members', type=int, default=5, help='total ensemble members (ICs) to generate')
parser.add_argument('--scale', type=str, default='1e-3', help='perturbation scale; also the output dir name')
parser.add_argument('--tf32', type=int, default=0, choices=(0, 1),
                    help='1 restores the pre-fix TF32 matmuls for a before/after precision comparison; 0 = true fp32')
parser.add_argument('--out_root', type=str, default='fp32', choices=('fp32', 'tf32'),
                    help='precision level of .../pangu/data/, so a --tf32 1 rerun cannot overwrite the fp32 data')
args = parser.parse_args()

model_numbers = [1,3,6,24]
hours = 72
scale = float(args.scale)

# Set the behavior of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False

options.intra_op_num_threads = 1
cuda_provider_options = {'device_id': args.device, 'arena_extend_strategy':'kSameAsRequested',
                         'use_tf32': str(args.tf32),}

ort_sessions = {}
def get_session(model_number):
    if model_number not in ort_sessions:
        ort_sessions.clear()
        gc.collect()
        ort_sessions[model_number] = ort.InferenceSession(f'models/pangu_weather_{model_number}.onnx',
                                                          sess_options=options,
                                                          providers=[('CUDAExecutionProvider', cuda_provider_options)])
    return ort_sessions[model_number]

input_data_dir = f'{pangu_scratch}/orig_data/oper_eda'
ensemble_mean_pl = np.load(os.path.join(input_data_dir, 'oper_eda_an_pl.npy')).mean(axis=1)
ensemble_mean_sfc = np.load(os.path.join(input_data_dir, 'oper_eda_an_sfc.npy')).mean(axis=1)

my_members = list(range(1, args.members + 1))[args.shard::args.num_shards]

for model_number in model_numbers:
    iterations = hours // model_number

    output_data_dir = f"{pangu_scratch}/data/{args.out_root}/autoreg/oper/E/{args.scale}/{model_number}h"
    os.makedirs(output_data_dir, exist_ok=True)

    for m in my_members:
        ensemble_dir = os.path.join(output_data_dir, f"{m}")
        os.makedirs(ensemble_dir, exist_ok=True)

        # [:, m] not [:, m-1]: array index IS the EDA number, so index 0 is the control
        pert = (np.load(os.path.join(input_data_dir, 'oper_eda_an_pl.npy'), mmap_mode='r')[:, m] - ensemble_mean_pl) * scale
        pert = pert.astype(np.float32)
        input_step = pert + ensemble_mean_pl
        input_step = input_step.astype(np.float32)

        pert_surface = (np.load(os.path.join(input_data_dir, 'oper_eda_an_sfc.npy'), mmap_mode='r')[:, m] - ensemble_mean_sfc) * scale
        pert_surface = pert_surface.astype(np.float32)
        input_surface_step = pert_surface + ensemble_mean_sfc
        input_surface_step = input_surface_step.astype(np.float32)

        np.save(os.path.join(ensemble_dir, f'output_upper_{model_number}_0'), input_step)
        np.save(os.path.join(ensemble_dir, f'output_surface_{model_number}_0'), input_surface_step)

        for t in range(1, iterations + 1):
            output, output_surface = get_session(model_number).run(None, {'input':input_step, 'input_surface':input_surface_step})
            np.save(os.path.join(ensemble_dir, f'output_upper_{model_number}_{t}'), output)
            np.save(os.path.join(ensemble_dir, f'output_surface_{model_number}_{t}'), output_surface)

            input_step, input_surface_step = output.astype(np.float32), output_surface.astype(np.float32)

        print(f"model {model_number}h member {m} done", flush=True)

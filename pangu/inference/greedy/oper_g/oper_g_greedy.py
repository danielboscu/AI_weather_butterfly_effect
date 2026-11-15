import os
import argparse
import numpy as np
import onnxruntime as ort
import gc

pangu_scratch = os.environ["pangu_scratch"]
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device_id this process binds to')
parser.add_argument('--shard', type=int, default=0, help='index of this worker among --num_shards')
parser.add_argument('--num_shards', type=int, default=1, help='total workers splitting the members across GPUs')
parser.add_argument('--scale', type=str, default='0.001', help='gaussian noise std in field units; also the output dir name')
parser.add_argument('--members', type=int, default=5, help='total ensemble members (ICs) to generate')
parser.add_argument('--tf32', type=int, default=0, choices=(0, 1),
                    help='1 restores the pre-fix TF32 matmuls for a before/after precision comparison; 0 = true fp32')
parser.add_argument('--out_root', type=str, default='fp32', choices=('fp32', 'tf32'),
                    help='precision level of .../pangu/data/, so a --tf32 1 rerun cannot overwrite the fp32 data')
args = parser.parse_args()

# Set the behavior of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False

options.intra_op_num_threads = 1
cuda_provider_options = {'device_id': args.device, 'arena_extend_strategy':'kSameAsRequested',
                         'use_tf32': str(args.tf32),}

pinned = (1,)
ort_sessions = {}
def get_session(model_step):
    if model_step in ort_sessions:
        return ort_sessions[model_step]
    sess = ort.InferenceSession(f'models/pangu_weather_{model_step}.onnx',
                                sess_options=options,
                                providers=[('CUDAExecutionProvider', cuda_provider_options)])
    if model_step in pinned:
        ort_sessions[model_step] = sess
    return sess


def get_times(hours):
    # get the times required a set number of hours
    times = []
    while hours > 0:
        if hours - 24 >= 0:
            times.append(24+times[-1] if times else 24)
            hours -= 24
        elif hours - 6 >= 0:
            times.append(6+times[-1] if times else 6)
            hours -= 6
        elif hours - 3 >= 0:
            times.append(3+times[-1] if times else 3)
            hours -= 3
        else:
            times.append(1+times[-1] if times else 1)
            hours -= 1
    return times

scale = float(args.scale)
input_data_dir = f'{pangu_scratch}/orig_data/oper_g'
output_data_dir = f"{pangu_scratch}/data/{args.out_root}/greedy/oper/G/{args.scale}"
os.makedirs(output_data_dir, exist_ok=True)
my_members = list(range(1, args.members + 1))[args.shard::args.num_shards]
for m in my_members:
    temp_path = os.path.join(output_data_dir, f"{m}")
    os.makedirs(temp_path, exist_ok=True)
    input_step = np.load(os.path.join(input_data_dir, 'oper_g_pl.npy'))
    noise = np.random.normal(0, scale, input_step.shape)
    input_step[2:, :, :, :] = (input_step[2:, :, :, :] + noise[2:, :, :, :]).astype(np.float32)

    input_surface_step = np.load(os.path.join(input_data_dir, 'oper_g_sfc.npy')).astype(np.float32)

    ic_upper = input_step
    ic_surface = input_surface_step
    np.save(os.path.join(temp_path, 'output_upper_greedy_0'), ic_upper)
    np.save(os.path.join(temp_path, 'output_surface_greedy_0'), ic_surface)

    for i in range(1,73):
        decomp = get_times(i)
        input_step = ic_upper
        input_surface_step = ic_surface
        prev_t = 0
        for t in decomp:
            model_step = t - prev_t
            upper_file = os.path.join(temp_path, f'{t}_{model_step}h_upper.npy')
            surface_file = os.path.join(temp_path, f'{t}_{model_step}h_surface.npy')
            if os.path.exists(upper_file):
                input_step = np.load(upper_file).astype(np.float32)
                input_surface_step = np.load(surface_file).astype(np.float32)
            else:
                output, output_surface = get_session(model_step).run(None, {'input':input_step, 'input_surface':input_surface_step})
                np.save(upper_file, output)
                np.save(surface_file, output_surface)
                input_step, input_surface_step = output.astype(np.float32), output_surface.astype(np.float32)
                if model_step not in pinned:
                    gc.collect()
            prev_t = t
